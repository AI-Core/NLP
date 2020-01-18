# %%
import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F
import math
from tqdm import tqdm
import os
import random

print("Hello. We are running")

# %%
SRC = Field(tokenize="spacy", tokenizer_language="de",
            init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize="spacy", tokenizer_language="en",
            init_token="<sos>", eos_token="<eos>", lower=True)
train_data, val_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(SRC, TRG))


# %%
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# %%
# device = "cpu"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if device == "cuda:1":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

BATCH_SIZE = 64

train_iter, val_iter, test_iter = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    repeat=True,
    shuffle=True
)

# %%
x_ = next(iter(train_iter))
toy_vocab = torch.Tensor([[1, 2, 3]]).long().to(device) # [a,b,c]
# %%
D_MODEL = 512
P_DROP = 0.1
NUM_HEADS = 8
D_FF = 2048
# %%


class Embeddings(nn.Module):
    def __init__(self, len_vocab, d_model=D_MODEL):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(len_vocab, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# %%
# toy_embedding_layer = Embeddings(toy_vocab.shape[-1]+1, d_model=4).to(device)
# toy_embeddings = toy_embedding_layer(toy_vocab)
# print(toy_embeddings, toy_embeddings.shape)

# %%


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, p_drop=P_DROP, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2)
        div_term = torch.pow(10000, (two_i/d_model)).float()
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)

        pe = pe.unsqueeze(0)

        # assigns the first argument to a class variable
        # i.e. self.pe
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(P_DROP)

    # x is the input embedding
    def forward(self, x):

        # work through this line :S
        x = x + self.pe[:, :x.size(1)].to(device)
        return self.dropout(x)


# %%
# toy_PE_layer = PositionalEncoding(d_model=4).to(device)
# toy_PEs = toy_PE_layer(toy_embeddings)
# print(toy_PEs)

# %%


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, p_drop=P_DROP):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads

        self.d_model = d_model
        self.num_heads = num_heads

        self.dropout = nn.Dropout(P_DROP)

        self.linear_Qs = [nn.Linear(d_model, self.d).to(device) for head in range(num_heads)]
        self.linear_Ks = [nn.Linear(d_model, self.d).to(device) for head in range(num_heads)]
        self.linear_Vs = [nn.Linear(d_model, self.d).to(device) for head in range(num_heads)]

        self.mha_linear = nn.Linear(d_model, d_model).to(device)

    def scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor, mask=None):
        Q_K_matmul = torch.matmul(Q, K.transpose(-2, -1))
        matmul_scaled = Q_K_matmul/math.sqrt(self.d)

        if mask is not None:
            matmul_scaled += (mask * '-inf')

        attention_weights = F.softmax(matmul_scaled, dim=-1)
        # print("WHAT's THE MASK", mask)
        # print("SCALED SOFTMAX", attention_weights)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, x: Tensor, queries: Tensor = None, keys: Tensor = None, values: Tensor = None, mask=None):

        q = x if not queries else queries
        if keys is not None:
            k = keys
        else:
            k = x

        if values is not None:
            v = values
        else:
            v = x

        # These will all be a list of Tensors
        Q = [linear(q) for linear in self.linear_Qs]
        K = [linear(k) for linear in self.linear_Ks]
        V = [linear(v) for linear in self.linear_Vs]

        # Why doesn't this work as expected?
        # scores_per_head, attention_weights = [self.scaled_dot_product_attention(
        #     Q_, K_, V_, mask) for (Q_, K_, V_) in zip(Q, K, V)]

        scores_per_head = []
        attention_weights_per_head = []
        for Q_, K_, V_ in zip(Q, K, V):
            score, attention_weight = self.scaled_dot_product_attention(
                Q_, K_, V_)
            scores_per_head.append(score)
            attention_weights_per_head.append(attention_weight)

        concat_scores = torch.cat(scores_per_head, -1)

        # shape: [B x num_head x S x S]
        attn_stacked = torch.stack(
            attention_weights_per_head, -1).permute(0, 3, 1, 2)

        return self.dropout(self.mha_linear(concat_scores)), attn_stacked


# %%
# toy_MHA_layer = MultiHeadAttention(d_model=4, num_heads=2).to(device)
# toy_MHA, attention_weights = toy_MHA_layer(toy_PEs)
# print(toy_MHA, toy_MHA.shape)


# %%
# temp_MHA_layer = MultiHeadAttention(d_model=3, num_heads=1)


# def print_out(q, k, v):
#     temp_out, temp_attn = temp_MHA_layer.scaled_dot_product_attention(
#         q, k, v, None)
#     print('Attention weights are:')
#     print(temp_attn)
#     print('Output is:')
#     print(temp_out)


# temp_k = torch.Tensor([[10, 0, 0],
#                        [0, 10, 0],
#                        [0, 0, 10],
#                        [0, 0, 10]])

# temp_v = torch.Tensor([[1, 0],
#                        [10, 0],
#                        [100, 5],
#                        [1000, 6]])

# temp_q = torch.Tensor([[0, 0, 10], [0, 10, 0], [10, 10, 0]])

# print_out(temp_q, temp_k, temp_v)

# temp_y = torch.rand((1, 60, 512))


# %%


class AddNorm(nn.Module):
    def __init__(self, d_model=D_MODEL, p_drop=P_DROP):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, res_input):
        ln = self.layer_norm(res_input + x)
        return self.dropout(ln)


# %%
# toy_AddNorm_layer = AddNorm(d_model=4).to(device)
# toy_AddNorm = toy_AddNorm_layer(toy_MHA, toy_PEs)
# print(toy_AddNorm, toy_AddNorm.shape)

# %%


class PointwiseFeedforward(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=D_FF, p_drop=P_DROP):
        super().__init__()
        self.pffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.pffn(x)


# %%
# toy_PFFN_layer = PointwiseFeedforward(d_model=4, d_ff=16).to(device)
# toy_PFFN = toy_PFFN_layer(toy_AddNorm)
# print(toy_PFFN, toy_PFFN.shape)

# %%
# toy_AddNorm_layer_2 = AddNorm(d_model=4).to(device)
# toy_AddNorm_2 = toy_AddNorm_layer_2(toy_PFFN, toy_AddNorm)
# print(toy_AddNorm_2, toy_AddNorm_2.shape)
# %%


class EncoderLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, p_drop=P_DROP):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_drop = p_drop

        self.MHA = MultiHeadAttention(
            self.d_model, self.num_heads, self.p_drop)

        self.addNorm1 = AddNorm(self.d_model, self.p_drop)
        self.addNorm2 = AddNorm(self.d_model, self.p_drop)

        self.PFFN = PointwiseFeedforward(
            self.d_model, self.d_ff, self.p_drop)

    def forward(self, x):
        mha, _ = self.MHA(x)
        addNorm_1 = self.addNorm1(mha, x)

        pffn = self.PFFN(addNorm_1)
        addNorm_2 = self.addNorm2(pffn, addNorm_1)

        return addNorm_2

# %%


class Encoder(nn.Module):
    def __init__(self, num_layers, len_vocab, d_model, num_heads, d_ff, p_drop, Embedding: Embeddings):
        super().__init__()

        self.len_vocab = len_vocab
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_drop = p_drop

        self.Embedding = Embedding

        self.PE = PositionalEncoding(
            self.d_model, self.p_drop)

        self.encoders = nn.ModuleList([EncoderLayer(
            self.d_model,
            self.num_heads,
            self.d_ff,
            self.p_drop
        ) for layer in range(num_layers)])
        self.encodersModelStack = nn.Sequential(*self.encoders)

    def forward(self, x):
        embeddings = self.Embedding(x)

        positional_encoding = self.PE(embeddings)
        return self.encodersModelStack(positional_encoding)


# %%
# toy_encoder = Encoder(3, 4, 4, 2, 16, 0.1, toy_embedding_layer).to(device)
# toy_encoder_output = toy_encoder(toy_vocab)
# print(toy_encoder_output, toy_encoder_output.shape)


# %%


class DecoderLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF, p_drop=P_DROP):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_drop = p_drop

        # These are both of type List<Tensors>.
        # They store the attention weights per layer
        # So if we had N decoder layers in our Decoder,
        # the final DecoderLayer will hold N attention weight tensors
        self.masked_mha_attn_weights = None
        self.enc_dec_mha_attn_weights = None

        self.addNorm1 = AddNorm(self.d_model, self.p_drop)
        self.addNorm2 = AddNorm(self.d_model, self.p_drop)
        self.addNorm3 = AddNorm(self.d_model, self.p_drop)

        self.MHA1 = MultiHeadAttention(
            self.d_model, self.num_heads, self.p_drop)
        self.MHA2 = MultiHeadAttention(
            self.d_model, self.num_heads, self.p_drop)

        self.PFFN = PointwiseFeedforward(
            self.d_model, self.d_ff, self.p_drop)

    def forward(self, inputs):
        x, encoder_output, mask, masked_mha_attn_weights, enc_dec_mha_attn_weights = inputs
        # add masking capabilities
        masked_mha, masked_mha_attn_weights = self.MHA1(x, mask=mask)
        addNorm_1 = self.addNorm1(masked_mha, x)

        mha, enc_dec_mha_attn_weights = self.MHA2(x, None, encoder_output, encoder_output)
        addNorm_2 = self.addNorm2(mha, addNorm_1)

        pffn = self.PFFN(addNorm_2)
        addNorm_3 = self.addNorm3(pffn, addNorm_2)

        self.masked_mha_attn_weights = masked_mha_attn_weights
        self.enc_dec_mha_attn_weights = enc_dec_mha_attn_weights

        return (addNorm_3, encoder_output, mask, self.masked_mha_attn_weights, self.enc_dec_mha_attn_weights)


# %%
class Decoder(nn.Module):
    def __init__(self, num_layers, len_vocab, d_model, num_heads, d_ff, p_drop, Embedding: Embeddings):
        super().__init__()

        self.len_vocab = len_vocab
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_drop = p_drop

        self.Embedding = Embedding

        self.PE = PositionalEncoding(
            self.d_model, self.p_drop)

        self.decoders = nn.ModuleList([DecoderLayer(
            self.d_model,
            self.num_heads,
            self.d_ff,
            self.p_drop
        ) for layer in range(num_layers)])

        self.decodersModelStack = nn.Sequential(*self.decoders)

    def create_mask(self, seq_len):
        ones_arr = torch.ones((seq_len, seq_len))
        mask = ones_arr.triu(1)
        return mask

    def forward(self, x, encoder_output):
        embeddings = self.Embedding(x)

        mask = self.create_mask(x.shape[1])

        positional_encoding = self.PE(embeddings)

        return self.decodersModelStack((positional_encoding, encoder_output, mask, None, None))


# %%
# toy_decoder = Decoder(3, 4, 4, 2, 16, 0.1, toy_embedding_layer).to(device)
# toy_decoder_output, _, _, toy_mmha_w, toy_e_d_mha_w = toy_decoder(
#     toy_vocab, toy_encoder_output)
# print(toy_decoder_output, toy_decoder_output.shape)
# print(toy_mmha_w, len(toy_mmha_w))
# print(toy_e_d_mha_w, len(toy_e_d_mha_w))

# %%


class Transformer(nn.Module):
    def __init__(self, num_layers, src_vocab_len, trg_vocab_len, d_model, num_heads, d_ff, p_drop):
        super().__init__()

        # len_vocab = src_vocab_len + trg_vocab_len

        encoder_Embedding = Embeddings(src_vocab_len, d_model)
        decoder_Embedding = Embeddings(trg_vocab_len, d_model)

        self.encoder = Encoder(num_layers, src_vocab_len,
                               d_model, num_heads, d_ff, p_drop, encoder_Embedding)
        self.decoder = Decoder(num_layers, trg_vocab_len,
                               d_model, num_heads, d_ff, p_drop, decoder_Embedding)

        # Maybe use target vocab size here
        self.linear_layer = nn.Linear(d_model, trg_vocab_len)

    def forward(self, input, target):
        encoder_outputs = self.encoder(input)
        decoder_output, _, _, masked_mha_attn_weights, enc_dec_mha_attn_weights = self.decoder(
            target, encoder_outputs)

        return (self.linear_layer(decoder_output), masked_mha_attn_weights, enc_dec_mha_attn_weights)


# %%
toy_transformer_layer = Transformer(
    2, 8000, 8500, 512, 8, 2048, 0.1
).to(device)

toy_input = torch.rand((64, 38)).long().to(device)
toy_target = torch.rand((64, 36)).long().to(device)
toy_output, _, _ = toy_transformer_layer(toy_input, toy_target)
print("TOY OUTPUT SHAPE:", toy_output.shape)
# print(toy_output)


# %%
def custom_lr_optimizer(optimizer: Adam, step, d_model=D_MODEL, warmup_steps=4000):
    min_arg1 = math.sqrt(1/(step+1))
    min_arg2 = step * (warmup_steps**-1.5)
    lr = math.sqrt(1/d_model) * min(min_arg1, min_arg2)

    optimizer.param_groups[0]["lr"] = lr

    return optimizer


# %%
def init_weights(model: nn.Module):
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

# %%
PAD_IDX = TRG.vocab.stoi["<pad>"]
src_vocab_len = SRC.vocab.__len__()
trg_vocab_len = TRG.vocab.__len__()
transformer = Transformer(6, src_vocab_len, trg_vocab_len, D_MODEL, NUM_HEADS, D_FF, P_DROP).to(device)

# %%
MODEL_PATH = "transformer_model.pt"
# %%
def train(STEPS=100000):
    EPOCHS = STEPS // len(train_iter)
    loss_list = []

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=PAD_IDX)
    optimizer = Adam(transformer.parameters(), betas=(0.9, 0.98))
    
    transformer.train()

    for step in tqdm(range(STEPS)):
        transformer.train()
        optimizer = custom_lr_optimizer(optimizer, step)
        optimizer.zero_grad()
        
        # Get current batch from train_iter.
        # Since we're shuffling with repeat, we just need
        # to call next(iter(.)) at every step
        t = next(iter(train_iter))
        t_src, t_trg = t.src.T, t.trg.T

        t_trg_inp = t_trg[:, :-1]
        t_trg_real = t_trg[:, 1:]

        predictions, _, _ = transformer(t_src, t_trg_inp)

        loss = criterion(predictions.permute(0, 2, 1), t_trg_real)
        loss.mean().backward()

        optimizer.step()

        loss_list.append(loss)

        if step % 50 == 0:
            print("Loss at {}th step: {}".format(step, loss.mean().item()))
            
            rand_index = random.randrange(BATCH_SIZE)

            transformer.eval()

            v = next(iter(test_iter))
            v_src, v_trg = v.src.T, v.trg.T
            v_trg_inp = v_trg[:, :-1]
            v_trg_real = v_trg[:, 1:]

            v_predictions, _, _ = transformer(v_src, v_trg_inp)
            max_args = v_predictions[rand_index].argmax(-1)
            print("For random element in TEST batch (real/pred)...")
            print(v_trg_real[rand_index, :])
            print(max_args)
            
            print("Length til first <PAD> (real -> pred)...")
            try:
                pred_PAD_idx = max_args.tolist().index(3)
            except:
                pred_PAD_idx = None

            print(v_trg_real[rand_index, :].tolist().index(3), "  --->  ", pred_PAD_idx)

        if step % 1000 == 0:
            torch.save(transformer, MODEL_PATH)


# %%
# %%
def evaluate():
    
    
    rand_index = random.randrange(BATCH_SIZE)

    # transformer.eval()

    v = next(iter(test_iter))
    v_src, v_trg = v.src.T, v.trg.T
    v_trg_inp = v_trg[:, :-1]
    v_trg_real = v_trg[:, 1:]

    v_predictions, _, _ = transformer(v_src, v_trg_inp)
    max_args = v_predictions[rand_index].argmax(-1)
    print("For random element in TEST batch (real/pred)...")
    print(v_trg_real[rand_index, :])
    print(max_args)    
    
    # transformer.to(device).eval()

    # test_data = next(iter(test_iter))
    # src, trg = test_data.src.T, test_data.trg.T
    # print(src[0])
    # print(trg[0])
    
    
    # src = torch.LongTensor([  2,   5, 842,   0, 149, 301,   4,   3,   1]).to(device)
    # trg = torch.LongTensor([   2,    4,  429, 4548,   51,   27,  394,   13,    4, 4642,    5,    3]).to(device)
    
    # src = src.unsqueeze(0)

    # pred = torch.LongTensor([2]).to(device)
    # pred = pred.unsqueeze(0)
    # print(pred.shape)

    # for i in range(40):
    #     predictions, _, _ = transformer(src, pred)
    #     predicted_id = predictions[:, -1:, :].argmax(-1)
    #     print("predicted id", predicted_id)
    #     if predicted_id.squeeze(0).item() == 3:
    #         break
    #     else:
    #         pred = torch.cat((pred, predicted_id), dim=-1)
    #         print("pred shape:", pred.shape)
    #         print("pred id shape:", predicted_id.shape)
    #         print("PRED:", pred)

    # src_tokens = [SRC.vocab.itos[i] for i in src.squeeze()]
    # trg_tokens = [TRG.vocab.itos[i] for i in trg.squeeze()]
    # pred_tokens = [TRG.vocab.itos[i] for i in pred.squeeze()]
    
    # print(src_tokens)
    # print(trg_tokens)
    # print(pred_tokens)


# %%
if not os.path.exists(MODEL_PATH):
    train()
else:
    transformer = torch.load(MODEL_PATH, map_location=torch.device(device))
    evaluate()