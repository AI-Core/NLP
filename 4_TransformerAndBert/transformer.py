from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from torchtext.data import BucketIterator, Field
from torchtext.datasets import Multi30k
import hyperparams as hp
import math as m
import copy
import numpy as np
from torch.autograd import Variable
from torch.optim.adam import Adam
from torch.nn.modules import CrossEntropyLoss
import random
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from torch import Tensor
import os
import sacrebleu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cuda:0":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

SRC = Field(tokenize="spacy", tokenizer_language="de",
            init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize="spacy", tokenizer_language="en",
            init_token="<sos>", eos_token="<eos>", lower=True)

train_data, val_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

train_iter, val_iter, test_iter = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=hp.BATCH_SIZE,
    device=device
)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model=hp.D_MODEL):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

    def forward(self, x):
        # shape(x) = [B x seq_len]

        embedding = self.embed(x)
        # shape(embedding) = [B x seq_len x D]

        return embedding * m.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, dropout=hp.P_DROP, max_seq_len=200):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model).to(device)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2).float()
        div_term = torch.pow(10000, (two_i/torch.Tensor([d_model]))).float()
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)

        pe = pe.unsqueeze(0)

        # assigns the first argument to a class variable
        # i.e. self.pe
        self.register_buffer("pe", pe)

    def forward(self, x):
        # shape(x) = [B x seq_len x D]
        pe = self.pe[:, :x.shape[1]].detach()
        x = x + pe
        # shape(x) = [B x seq_len x D]
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, dropout=hp.P_DROP):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # shape(x) = [B x seq_len x D]
        ln = self.layer_norm(x)
        # shape(ln) = [B x seq_len x D]
        return self.dropout(ln)


class PWFFN(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, d_ff=hp.D_FF, dropout=hp.P_DROP):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # shape(x) = [B x seq_len x D]

        ff = self.ff(x)
        # shape(ff) = [B x seq_len x D]

        return self.ff(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, num_heads=hp.HEADS, dropout=0.1):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads

        self.d_model = d_model
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)

        self.linear_Qs = [nn.Linear(d_model, self.d).to(device)
                          for _ in range(num_heads)]
        self.linear_Ks = [nn.Linear(d_model, self.d).to(device)
                          for _ in range(num_heads)]
        self.linear_Vs = [nn.Linear(d_model, self.d).to(device)
                          for _ in range(num_heads)]

        self.mha_linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, mask=None):
        # shape(Q) = [B x seq_len (Q) x D/num_heads]
        # shape(K, V) = [B x seq_len (K, V) x D/num_heads]

        Q_K_matmul = torch.matmul(Q, K.permute(0, 2, 1))
        scores = Q_K_matmul/m.sqrt(self.d)
        # shape(scores) = [B x ??_seq_len x SRC_seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        # shape(attention_weights) = [B x seq_len (K, V) x seq_len (Q)]

        output = torch.matmul(attention_weights, V)
        # shape(output) = [B x seq_len (K, V) x D/num_heads]

        return output, attention_weights

    def forward(self, pre_q, pre_k, pre_v, mask=None):
        # shape(pre_q (ENCODING)) = [B x SRC_seq_len x D]
        # shape(pre_q (DECODING)) = [B x TRG_seq_len x D]
        #
        # shape(pre_k, pre_v (MASKED ATTENTION)) = [B x TRG_seq_len x D]
        # shape(pre_k, pre_v (OTHERWISE)) = [B x SRC_seq_len x D]

        Q = [linear_Q(pre_q) for linear_Q in self.linear_Qs]
        K = [linear_K(pre_k) for linear_K in self.linear_Ks]
        V = [linear_V(pre_v) for linear_V in self.linear_Vs]
        # shape(Q) = [B x seq_len (Q) x D/num_heads] * num_heads
        # shape(K) = [B x seq_len (K, V) x D/num_heads] * num_heads
        # shape(V) = [B x seq_len (K, V) x D/num_heads] * num_heads

        output_per_head = []
        attn_weights_per_head = []
        # shape(output_per_head) = [B x seq_len (K, V) x D/num_heads] * num_heads
        # shape(attn_weights_per_head) = [B x seq_len (K, V) x seq_len (Q)] * num_heads
        for Q_, K_, V_ in zip(Q, K, V):
            output, attn_weight = self.scaled_dot_product_attention(
                Q_, K_, V_, mask)
            output_per_head.append(output)
            attn_weights_per_head.append(attn_weight)

        output = torch.cat(output_per_head, -1)
        attn_weights = torch.stack(attn_weights_per_head).permute(0, 3, 1, 2)
        # shape(output) = [B x seq_len (K, V) x D]
        # shape(attn_weights) = [B x num_heads x seq_len (K, V) x seq_len(Q)]

        return self.mha_linear(self.dropout(output)), attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, num_heads=hp.HEADS, d_ff=hp.D_FF, dropout=hp.P_DROP):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = PWFFN(d_model, d_ff)

    def forward(self, x, mask):
        # shape(x) = [B x seq_len x D]

        mha, encoder_attention_weights = self.mha(x, x, x, mask)
        norm1 = self.norm_1(x + mha)
        # shape(mha) = [B x seq_len x D]
        # shape(encoder_attention_weights) = [B x num_heads x seq_len x seq_len]
        # shape(norm1) = [B x seq_len x D]

        ff = self.ff(norm1)
        norm2 = self.norm_2(norm1 + ff)
        # shape(ff) = [B x seq_len x D]
        # shape(norm2) = [B x seq_len x D]

        return norm2, encoder_attention_weights


class Encoder(nn.Module):
    def __init__(self, Embedding: Embeddings, d_model=hp.D_MODEL, num_heads=hp.HEADS, num_layers=hp.LAYERS, d_ff=hp.D_FF, dropout=hp.P_DROP):
        super().__init__()

        self.Embedding = Embedding

        self.PE = PositionalEncoding(
            d_model)

        self.encoders = nn.ModuleList([copy.deepcopy(EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )) for layer in range(num_layers)])

    def forward(self, x, mask):
        # shape(x) = [B x SRC_seq_len]

        embeddings = self.Embedding(x)
        encoding = self.PE(embeddings)
        # shape(embeddings) = [B x SRC_seq_len x D]
        # shape(encoding) = [B x SRC_seq_len x D]

        for encoder in self.encoders:
            encoding, encoder_attention_weights = encoder(encoding, mask)
            # shape(encoding) = [B x SRC_seq_len x D]
            # shape(encoder_attention_weights) = [B x SRC_seq_len x SRC_seq_len]

        return encoding, encoder_attention_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, num_heads=hp.HEADS, d_ff=hp.D_FF, dropout=hp.P_DROP):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.mha_2 = MultiHeadAttention(d_model, num_heads)
        self.ff = PWFFN(d_model, d_ff)

    def forward(self, x, encoder_outputs, trg_mask, src_mask):
        # shape(x) = [B x TRG_seq_len x D]
        # shape(encoder_outputs) = [B x SRC_seq_len x D]

        masked_mha, masked_mha_attn_weights = self.mha_1(
            x, x, x, mask=trg_mask)
        # shape(masked_mha) = [B x TRG_seq_len x D]
        # shape(masked_mha_attn_weights) = [B x num_heads x TRG_seq_len x TRG_seq_len]

        norm1 = self.norm_1(x + masked_mha)
        # shape(norm1) = [B x TRG_seq_len x D]

        enc_dec_mha, enc_dec_mha_attn_weights = self.mha_2(
            norm1, encoder_outputs, encoder_outputs, mask=src_mask)
        # shape(enc_dec_mha) = [B x TRG_seq_len x D]
        # shape(enc_dec_mha_attn_weights) = [B x num_heads x TRG_seq_len x SRC_seq_len]

        norm2 = self.norm_2(norm1 + enc_dec_mha)
        # shape(norm2) = [B x TRG_seq_len x D]

        ff = self.ff(norm2)
        norm3 = self.norm_3(norm2 + ff)
        # shape(ff) = [B x TRG_seq_len x D]
        # shape(norm3) = [B x TRG_seq_len x D]

        return norm3, masked_mha_attn_weights, enc_dec_mha_attn_weights


class Decoder(nn.Module):
    def __init__(self, Embedding: Embeddings, d_model=hp.D_MODEL, num_heads=hp.HEADS, num_layers=hp.LAYERS, d_ff=hp.D_FF, dropout=hp.P_DROP):
        super().__init__()

        self.Embedding = Embedding

        self.PE = PositionalEncoding(
            d_model)

        self.decoders = nn.ModuleList([copy.deepcopy(DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )) for layer in range(num_layers)])

    def forward(self, x, encoder_output, trg_mask, src_mask):
        # shape(x) = [B x TRG_seq_len]

        embeddings = self.Embedding(x)
        # shape(embeddings) = [B x TRG_seq_len x D]

        encoding = self.PE(embeddings)
        for decoder in self.decoders:
            encoding, masked_mha_attn_weights, enc_dec_mha_attn_weights = decoder(
                encoding, encoder_output, trg_mask, src_mask)
            # shape(encoding) = [B x num_heads x TRG_seq_len x SRC_seq_len]
            # shape(masked_mha_attn_weights) = [B x num_heads x TRG_seq_len x TRG_seq_len]
            # shape(enc_dec_mha_attn_weights) = [B x num_heads x TRG_seq_len x SRC_seq_len]

        return encoding, enc_dec_mha_attn_weights, masked_mha_attn_weights


class Transformer(nn.Module):
    def __init__(self, src_vocab_len, trg_vocab_len, d_model=hp.D_MODEL, d_ff=hp.D_FF, num_layers=hp.LAYERS, num_heads=hp.HEADS, dropout=hp.P_DROP):
        super().__init__()

        self.num_heads = num_heads

        encoder_Embedding = Embeddings(
            src_vocab_len, SRC.vocab.stoi["<pad>"], d_model)
        decoder_Embedding = Embeddings(
            trg_vocab_len, TRG.vocab.stoi["<pad>"], d_model)

        self.encoder = Encoder(encoder_Embedding, d_model,
                               num_heads, num_layers, d_ff, dropout)
        self.decoder = Decoder(decoder_Embedding, d_model,
                               num_heads, num_layers, d_ff, dropout)

        self.linear_layer = nn.Linear(d_model, trg_vocab_len)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_src_mask(self, src):
        src_mask = (src != SRC.vocab.stoi["<pad>"]).unsqueeze(-2)
        return src_mask

    def create_trg_mask(self, trg):
        trg_mask = (trg != TRG.vocab.stoi["<pad>"]).unsqueeze(-2)
        size = trg.shape[1]  # get seq_len for matrix
        mask = torch.ones((1, size, size)).triu(1).to(device)
        mask = mask == 0
        trg_mask = trg_mask & mask
        return trg_mask

    def forward(self, src, trg):
        # shape(src) = [B x SRC_seq_len]
        # shape(trg) = [B x TRG_seq_len]

        src_mask = self.create_src_mask(src)
        trg_mask = self.create_trg_mask(trg)
        # shape(src_mask) = [B x 1 x SRC_seq_len]
        # shape(trg_mask) = [B x 1 x TRG_seq_len]

        encoder_outputs, encoder_mha_attn_weights = self.encoder(src, src_mask)
        # shape(encoder_outputs) = [B x SRC_seq_len x D]
        # shape(encoder_mha_attn_weights) = [B x num_heads x SRC_seq_len x SRC_seq_len]
        decoder_outputs, enc_dec_mha_attn_weights, masked_mha_attn_weights = self.decoder(
            trg, encoder_outputs, trg_mask, src_mask)
        # shape(decoder_outputs) = [B x SRC_seq_len x D]
        # shape(enc_dec_mha_attn_weights) = [B x num_heads x TRG_seq_len x SRC_seq_len]
        logits = self.linear_layer(decoder_outputs)
        # shape(logits) = [B x TRG_seq_len x TRG_vocab_size]

        return logits, encoder_mha_attn_weights, enc_dec_mha_attn_weights, masked_mha_attn_weights


def custom_lr_optimizer(optimizer: Adam, step, d_model=hp.D_MODEL, warmup_steps=4000):
    min_arg1 = m.sqrt(1/(step+1))
    min_arg2 = step * (warmup_steps**-1.5)
    lr = m.sqrt(1/d_model) * min(min_arg1, min_arg2)

    optimizer.param_groups[0]["lr"] = lr

    return optimizer


def train(model, SRC, TRG, MODEL_PATH, FORCE_MAX_LEN=50):
    model.train()
    optimizer = Adam(
        model.parameters(), lr=hp.LR, betas=(0.9, 0.98), eps=1e-9)
    criterion = CrossEntropyLoss(ignore_index=TRG.vocab.stoi["<pad>"])

    for epoch in tqdm(range(hp.EPOCHS)):

        for step, batch in enumerate(train_iter):
            global_step = epoch * len(train_iter) + step

            model.train()
            optimizer.zero_grad()
            optimizer = custom_lr_optimizer(optimizer, global_step)

            src = batch.src.T
            trg = batch.trg.T

            trg_input = trg[:, :-1]

            preds, _, _, _ = model(src, trg_input)
            ys = trg[:, 1:]

            loss = criterion(preds.permute(0, 2, 1), ys)
            loss.mean().backward()
            optimizer.step()

            if global_step % 50 == 0:
                print("#"*90)

                rand_index = random.randrange(hp.BATCH_SIZE)

                model.eval()

                v = next(iter(val_iter))
                v_src, v_trg = v.src.T, v.trg.T

                v_trg_inp = v_trg[:, :-1].detach()
                v_trg_real = v_trg[:, 1:].detach()

                v_predictions, _, _, _ = model(v_src, v_trg_inp)
                max_args = v_predictions[rand_index].argmax(-1)
                print("For random element in VALIDATION batch (real/pred)...")
                print([TRG.vocab.itos[word_idx]
                       for word_idx in v_trg_real[rand_index, :]])
                print([TRG.vocab.itos[word_idx]
                       for word_idx in max_args])

                print("Length til first <PAD> (real -> pred)...")
                try:
                    pred_PAD_idx = max_args.tolist().index(3)
                except:
                    pred_PAD_idx = None

                print(v_trg_real[rand_index, :].tolist().index(
                    3), "  --->  ", pred_PAD_idx)

                val_loss = criterion(
                    v_predictions.permute(0, 2, 1), v_trg_real)
                print("TRAINING LOSS:", loss.mean().item())
                print("VALIDATION LOSS:", val_loss.mean().item())

                print("#"*90)

                writer.add_scalar(
                    "Training Loss", loss.mean().detach().item(), global_step)
                writer.add_scalar("Validation Loss",
                                  val_loss.mean().detach().item(), global_step)
        torch.save(model, MODEL_PATH)


def search(model, source_sentences, src_sos_idx=SRC.vocab.stoi["<sos>"], trg_sos_idx=TRG.vocab.stoi["<sos>"], max_seq_len=40):
    src = source_sentences.to(device)
    # shape(src) = [B x seq_len]

    batch_size = src.shape[0]
    seq_len = src.shape[1]

    outputs = torch.zeros(batch_size, max_seq_len).long().to(device)

    for seq_id in range(batch_size):
        input_sequence = src[seq_id].unsqueeze(0)
        preds = torch.LongTensor([trg_sos_idx]).to(device).unsqueeze(0)

        for t in range(max_seq_len-1):
            predictions, _, _, _ = transformer(input_sequence, preds)
            predicted_id = predictions[:, -1:, :].argmax(-1)
            preds = torch.cat((preds, predicted_id), dim=-1)

        outputs[seq_id] = preds

    return outputs


def get_text_from_tensor(tensor, SRC_or_TRG):
    # shape(tensor) = [B x seq_len]
    batch_output = []

    sos = SRC_or_TRG.vocab.stoi["<sos>"]
    eos = SRC_or_TRG.vocab.stoi["<eos>"]
    pad = SRC_or_TRG.vocab.stoi["<pad>"]

    for i in range(tensor.shape[0]):
        sequence = tensor[i]
        words = []
        for tok_idx in sequence:
            tok_idx = int(tok_idx)
            token = SRC_or_TRG.vocab.itos[tok_idx]

            if token == sos:
                continue
            elif token == eos or token == pad:
                break
            else:
                words.append(token)
        words = " ".join(words)
        batch_output.append(words)
    return batch_output


def evaluate_bleu(model, iterator):

    model.eval()

    hyp = []
    ref = []

    for batch in tqdm(iterator):
        src, trg = batch.src.T, batch.trg.T
        outputs = search(model, src)

        outputs = outputs[:, 1:]

        hyp += get_text_from_tensor(outputs, TRG)
        ref += get_text_from_tensor(trg, TRG)

    # expand dim of reference list
    # sys = ['translation_1', 'translation_2']
    # ref = [['truth_1', 'truth_2'], ['another truth_1', 'another truth_2']]
    ref = [ref]
    return sacrebleu.corpus_bleu(hyp, ref, force=True).score


def inference(model, source_sentence):
    source_sentence_tokens = SRC.preprocess(source_sentence)
    src = SRC.process([source_sentence_tokens]).T
    outputs = search(model, src)
    print(get_text_from_tensor(outputs, TRG))


if __name__ == "__main__":
    MODEL_PATH = "transformer_model.pt"
    if not os.path.exists(MODEL_PATH):
        writer = SummaryWriter()
        transformer = Transformer(len(SRC.vocab), len(TRG.vocab)).to(device)
        train(transformer, SRC, TRG, MODEL_PATH)
    else:
        transformer = torch.load(MODEL_PATH, map_location=device)
        inference(transformer, "Eine Frau mit blonden Haaren trinkt aus einem Glas")
        print(evaluate_bleu(transformer, test_iter))
