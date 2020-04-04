
import random
from torch import nn, optim
import torch
import os
import math as m
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data.dataset import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cuda:0":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class BertEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model, max_len=512, p_drop=0.1):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.seg_embedding = nn.Embedding(3, d_model, padding_idx=0)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, seg_ids):
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, dtype=torch.long).to(device)

        tok_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(pos_ids)
        seg_emb = self.seg_embedding(seg_ids)

        embeddings = tok_emb + pos_emb + seg_emb
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Norm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # shape(x) = [B x seq_len x D]
        ln = self.layer_norm(x)
        # shape(ln) = [B x seq_len x D]
        return self.dropout(ln)


class PWFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
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
    def __init__(self, d_model, num_heads, dropout):
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
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model, dropout=dropout)
        self.norm_2 = Norm(d_model, dropout=dropout)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = PWFFN(d_model, d_ff, dropout=dropout)

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


class BertModel(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, p_drop=0.1):
        super().__init__()

        self.embedding = BertEmbedding(vocab_size, d_model, max_len, p_drop)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, p_drop) for _ in range(num_layers)])

    def forward(self, x, seg_ids, mask):
        x = self.embedding(x, seg_ids)

        for layer in self.layers:
            x, _ = layer(x, mask)

        return x


class BERT(nn.Module):
    def __init__(self, bert, vocab_size, d_model, pad_idx=0):
        super().__init__()

        self.pad_idx = pad_idx
        self.bert = bert
        self.mlm_layer = nn.Linear(d_model, vocab_size)
        self.nsp_layer = nn.Linear(d_model, 1)

    def create_padding_mask(self, x):
        mask = (x != self.pad_idx).unsqueeze(-2)
        return mask.to(device)

    def forward(self, x, seg_ids):
        mask = self.create_padding_mask(x)

        hidden = self.bert(x, seg_ids, mask)

        mlm_logits = self.mlm_layer(hidden)
        nsp_logits = self.nsp_layer(hidden[:, 0])

        return mlm_logits, nsp_logits


class Wiki2BertDataset(Dataset):
    def __init__(self, path, max_len):
        with open(path, 'r') as f:
            data = f.readlines()
        self.corpus = []
        for line in data:
            if len(line.strip()) > 0:
                self.corpus += line.strip().lower().split(" . ")

        self.vocab = set()
        for line in self.corpus:
            for word in line.split():
                self.vocab.add(word)
        self.vocab = ['<PAD>', '<CLS>', '<SEP>', '<MASK>'] + list(self.vocab)

        self.stoi = {w: idx for (idx, w) in enumerate(self.vocab)}
        self.itos = {idx: w for (idx, w) in enumerate(self.vocab)}

        self.max_len = max_len

    def __getitem__(self, index):
        s1 = self.corpus[index-1].strip()
        s2 = self.corpus[index].strip()
        is_next = 1

        # 50% next sentence, 50% random sentence
        if random.random() < 0.5:
            random_idx = random.randrange(len(self.corpus))
            s2 = self.corpus[random_idx].strip()
            is_next = 0

        # add masking
        s1 = s1.split()
        s2 = s2.split()
        s1_mask, s1 = self.mask_sentence(s1)
        s2_mask, s2 = self.mask_sentence(s2)

        s1_mask = [self.stoi['<CLS>']] + s1_mask + [self.stoi['<SEP>']]
        s2_mask = s2_mask + [self.stoi['<SEP>']]

        s1 = [self.stoi['<CLS>']] + s1 + [self.stoi['<PAD>']]
        s2 = s2 + [self.stoi['<PAD>']]

        masked_sequence = (s1_mask + s2_mask)[:self.max_len]

        raw_sequence = (s1 + s2)[:self.max_len]
        seg_ids = ([1 for _ in range(len(s1))] +
                   [2 for _ in range(len(s2))])[:self.max_len]

        padding = [self.stoi['<PAD>']
                   for _ in range(self.max_len - len(raw_sequence))]
        masked_sequence.extend(padding)
        raw_sequence.extend(padding)
        seg_ids.extend(padding)

        data = {"masked": masked_sequence,
                "raw": raw_sequence,
                "segment": seg_ids,
                "is_next": is_next}

        return {key: torch.tensor(value) for key, value in data.items()}

    def mask_sentence(self, sentence):
        masked_ids = []
        origninal_ids = []

        for i, token in enumerate(sentence):
            # select 15% tokens to mask
            origninal_ids.append(self.stoi[token])

            if random.random() < 0.15:
                mask_prob = random.random()

                # 80% prob to mask
                if mask_prob < 0.8:
                    masked_ids.append(self.stoi['<MASK>'])
                # 10% prob replace with random token
                elif mask_prob < 0.9:
                    masked_ids.append(random.randrange(len(self.vocab)))
                # 10% prob keep raw token
                else:
                    masked_ids.append(self.stoi[token])
            else:
                masked_ids.append(self.stoi[token])

        return masked_ids, origninal_ids

    def __len__(self):
        return len(self.corpus)


MAX_LEN = 64
BATCH_SIZE = 16


train_data = Wiki2BertDataset("train.txt", MAX_LEN)
test_data = Wiki2BertDataset("test.txt", MAX_LEN)
train_data_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, num_workers=4)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=4)

VOCAB_SIZE = len(train_data.vocab)
D_MODEL = 512
N_HEADS = 8
D_FF = 2048
N_LAYERS = 6


def train(model, data_loader, mlm_criterion, nsp_criterion, optimizer):
    model.train()

    total_loss = 0
    total_correct = 0
    total_element = 0
    for i, batch in enumerate(data_loader):

        X = batch['masked'].to(device)
        Y = batch['raw'].to(device)

        seg_ids = batch['segment'].to(device)
        is_next = batch['is_next'].to(device)

        optimizer.zero_grad()

        mlm_logits, nsp_logits = model(X, seg_ids)

        mlm_loss = mlm_criterion(mlm_logits.transpose(1, 2), Y)
        nsp_loss = nsp_criterion(nsp_logits.squeeze(), is_next.float())

        correct = nsp_logits.squeeze().round().eq(is_next).sum().item()
        element = is_next.nelement()
        total_correct += correct
        total_element += element

        loss = mlm_loss + nsp_loss

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Batch:\t {0} / {1},\t loss: {2:2.3f}, \t nsp_acc: {3:3.1f}%'.format(
                i, len(data_loader), loss.item(), total_correct*100/total_element))

        total_loss += loss.item()

    return total_loss / len(data_loader), total_correct*100 / total_element


def eval(model, data_loader, mlm_criterion, nsp_criterion):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_element = 0

    with torch.no_grad():
        for _, batch in enumerate(data_loader):

            X = batch['masked'].to(device)
            Y = batch['raw'].to(device)

            seg_ids = batch['segment'].to(device)
            is_next = batch['is_next'].to(device)

            mlm_logits, nsp_logits = model(X, seg_ids)

            mlm_loss = mlm_criterion(mlm_logits.transpose(1, 2), Y)
            nsp_loss = nsp_criterion(nsp_logits.squeeze(), is_next.float())

            correct = nsp_logits.squeeze().round().eq(is_next).sum().item()

            element = is_next.nelement()
            total_correct += correct
            total_element += element

            loss = mlm_loss + nsp_loss

            total_loss += loss.item()

            # RANDOM TEST EXAMPLE:
            rand_index = random.randrange(BATCH_SIZE)

            output = mlm_logits[rand_index].argmax(dim=-1).squeeze(0)

            raw_v_X = [train_data.itos[idx]
                       for idx in output.cpu().numpy() if idx != 0]
            raw_v_Y = [train_data.itos[idx]
                       for idx in Y[rand_index].cpu().numpy() if idx != 0]

    return total_loss / len(data_loader), total_correct*100 / total_element, raw_v_X, raw_v_Y


EPOCH = 10


def train_loop(model, MODEL_PATH):

    best_loss = float('inf')
    optimizer = optim.Adam(model.parameters())
    mlm_criterion = nn.CrossEntropyLoss(
        ignore_index=0)
    nsp_criterion = nn.BCEWithLogitsLoss()

    for i in range(EPOCH):
        print('Start training Epoch {}:'.format(i+1))
        # train_loss, train_nsp_acc = train(
        #     model, train_data_loader, mlm_criterion, nsp_criterion, optimizer)

        test_loss, test_nsp_acc, raw_v_X, raw_v_Y = eval(
            model, test_data_loader, mlm_criterion, nsp_criterion)

        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     torch.save(model, MODEL_PATH)

        # print('Epoch {0} Train loss: {1:.3f} | Train acc: {2:3.1f}%'.format(
        #     i+1, train_loss, train_nsp_acc))
        print('Epoch {0} Test loss: {1:.3f} | Test acc: {2:3.1f}%'.format(
            i+1, test_loss, test_nsp_acc))
        print("#" * 90)
        print("For RANDOM example in test batch...")
        print(raw_v_X)
        print()
        print(raw_v_Y)
        print("#" * 90)


def predict_masked_sequence(masked_sequence, model, dataset):
    sequence = masked_sequence.strip().split()
    sequence_ids = [dataset.stoi[word] for word in sequence]

    sequence_ids = [dataset.stoi['<CLS>']] + \
        sequence_ids + [dataset.stoi['<SEP>']]
    seg_ids = [1 for _ in range(len(sequence_ids))]

    padding = [dataset.stoi['<PAD>']
               for _ in range(dataset.max_len - len(sequence_ids))]
    sequence_ids.extend(padding)
    seg_ids.extend(padding)

    input_sequence = torch.tensor(sequence_ids)
    input_segment = torch.tensor(seg_ids)

    input_sequence = input_sequence.unsqueeze(0).to(device)
    input_segment = input_segment.unsqueeze(0).to(device)

    with torch.no_grad():
        mlm_logits, _ = model(input_sequence, input_segment)

    output = mlm_logits.argmax(dim=-1).squeeze(0)
    output_text = [dataset.itos[idx]
                   for idx in output.cpu().numpy() if idx != 0]

    print(masked_sequence)
    print(" ".join(output_text))


if __name__ == "__main__":
    MODEL_PATH = "bert_model.pt"
    if not os.path.exists(MODEL_PATH):
        bert = BertModel(VOCAB_SIZE, D_MODEL, N_HEADS,
                         D_FF, N_LAYERS, MAX_LEN).to(device)
        model = BERT(bert, VOCAB_SIZE, D_MODEL, 0).to(device)
        train_loop(model, MODEL_PATH)
    else:
        model = torch.load(MODEL_PATH)
        predict_masked_sequence(
            "he went to the <MASK> and bought <MASK> .", model, train_data)
