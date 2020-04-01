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
        return self.embed(x) * m.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, dropout=hp.P_DROP, max_seq_len=200):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model).to(device)
        pos = torch.arange(0, max_seq_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2).float()
        div_term = torch.pow(10000, (two_i/d_model)).float()
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)

        pe = pe.unsqueeze(0)

        # assigns the first argument to a class variable
        # i.e. self.pe
        self.register_buffer("pe", pe)

    def forward(self, x):
        # add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len].detach()
        x = x + pe
        return self.dropout(x)


class Norm(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, dropout=hp.P_DROP):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        ln = self.layer_norm(x)
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
        # shape(Q) = [B * num_heads x ??_seq_len x D/num_heads]
        # shape(K = V (DECODING)) = [B * num_heads x SRC_seq_len x D/num_heads]
        # shape(mask) = [B * num_heads x TRG_seq_len x TRG_seq_len]

        # shape(QK_matmul) = [B * num_heads x ??_seq_len x SRC_seq_len]
        # shape(QK_matmul) = [B x num_heads x ??_seq_len x SRC_seq_len]
        Q_K_matmul = torch.matmul(Q, K.transpose(-2, -1))
        # Q_K_matmul = torch.matmul(Q, K.permute(0, 2, 1))
        matmul_scaled = Q_K_matmul/m.sqrt(self.d)

        if mask is not None:
            ##############
            # mask = mask.unsqueeze(1)
            matmul_scaled = matmul_scaled.masked_fill(mask == 0, -1e9)

        # shape(attention_weights) = [B * num_heads x ??_seq_len x SRC_seq_len]
        attention_weights = F.softmax(matmul_scaled, dim=-1)

        # shape(output) = [B * num_heads x ??_seq_len x D/num_heads]
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, q, k, v, mask=None):

        # shape(Q) = [B x ??_seq_len x D]
        # shape(K = V) = IF DECODER: [B x SRC_seq_len x D]
        # Q = self.linear_Q(q)
        # K = self.linear_K(k)
        # V = self.linear_V(v)

        Q = [linear_Q(q) for linear_Q in self.linear_Qs]
        K = [linear_K(k) for linear_K in self.linear_Ks]
        V = [linear_V(v) for linear_V in self.linear_Vs]

        batch_size = q.size(0)
        seq_len = q.size(1)

        scores_per_head = []
        aw_per_head = []
        for Q_, K_, V_ in zip(Q, K, V):
            scores, aw = self.scaled_dot_product_attention(Q_, K_, V_, mask)
            scores_per_head.append(scores)
            aw_per_head.append(aw)

        scores = torch.cat(scores_per_head, -1)
        aws = torch.stack(aw_per_head)

        return self.mha_linear(self.dropout(scores)), aws


class EncoderLayer(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, num_heads=hp.HEADS, d_ff=hp.D_FF, dropout=hp.P_DROP):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = PWFFN(d_model, d_ff)

    def forward(self, x, mask):
        mha, _ = self.mha(x, x, x, mask)
        norm1 = self.norm_1(x + mha)

        ff = self.ff(norm1)
        norm2 = self.norm_1(norm1 + ff)

        return norm2


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
        embeddings = self.Embedding(x)
        encoding = self.PE(embeddings)

        # encoding, mask = self.encodersModelStack((encoding, mask))
        for encoder in self.encoders:
            encoding = encoder(encoding, mask)

        return encoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model=hp.D_MODEL, num_heads=hp.HEADS, d_ff=hp.D_FF, dropout=hp.P_DROP):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.mha_2 = MultiHeadAttention(d_model, num_heads)
        self.ff = PWFFN(d_model, d_ff)

    def forward(self, x, e_outputs, trg_mask, src_mask):

        masked_mha, masked_mha_attn_weights = self.mha_1(
            x, x, x, mask=trg_mask)
        norm1 = self.norm_1(x + masked_mha)

        enc_dec_mha, enc_dec_mha_attn_weights = self.mha_2(
            norm1, e_outputs, e_outputs, mask=src_mask)
        norm2 = self.norm_2(norm1 + enc_dec_mha)

        ff = self.ff(norm2)
        norm3 = self.norm_3(norm2 + ff)

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
        embeddings = self.Embedding(x)

        masked_mha_attn_weights = None
        enc_dec_mha_attn_weights = None

        encoding = self.PE(embeddings)
        for decoder in self.decoders:
            encoding, masked_mha_attn_weights, enc_dec_mha_attn_weights = decoder(
                encoding, encoder_output, trg_mask, src_mask)

        return encoding, masked_mha_attn_weights, enc_dec_mha_attn_weights


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
        # return src_mask.repeat(self.num_heads, 1, 1)
        return src_mask

    def create_trg_mask(self, trg):
        trg_mask = (trg != TRG.vocab.stoi["<pad>"]).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        mask = torch.ones((1, size, size)).triu(1).to(device)
        mask = mask == 0
        trg_mask = trg_mask & mask
        return trg_mask
        # return trg_mask.repeat(self.num_heads, 1, 1)

    def forward(self, src, trg):
        src_mask = self.create_src_mask(src)
        trg_mask = self.create_trg_mask(trg)

        e_outputs = self.encoder(src, src_mask)
        d_output, _, _ = self.decoder(trg, e_outputs, trg_mask, src_mask)
        logits = self.linear_layer(d_output)

        return logits


def custom_lr_optimizer(optimizer: Adam, step, d_model=hp.D_MODEL, warmup_steps=4000):
    min_arg1 = m.sqrt(1/(step+1))
    min_arg2 = step * (warmup_steps**-1.5)
    lr = m.sqrt(1/d_model) * min(min_arg1, min_arg2)

    optimizer.param_groups[0]["lr"] = lr

    return optimizer


def train(model, SRC, TRG, FORCE_MAX_LEN=50, MODEL_PATH="transformer_model.pt"):
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

            preds = model(src, trg_input)
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
                if v_src.shape[1] > FORCE_MAX_LEN:
                    v_src = v_src[:, :FORCE_MAX_LEN]
                    v_src[:, FORCE_MAX_LEN-1] = SRC.vocab.stoi["<eos>"]
                if v_trg.shape[1] > FORCE_MAX_LEN:
                    v_trg = v_trg[:, :FORCE_MAX_LEN]
                    v_trg[:, FORCE_MAX_LEN-1] = TRG.vocab.stoi["<eos>"]

                v_trg_inp = v_trg[:, :-1].detach()
                v_trg_real = v_trg[:, 1:].detach()

                v_predictions = model(v_src, v_trg_inp)
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


if __name__ == "__main__":
    writer = SummaryWriter()
    model = Transformer(len(SRC.vocab), len(TRG.vocab)).to(device)
    train(model, SRC, TRG)
