import os
import torch
from torchtext import data
# We will work with a dataset from the torchtext package consists of data processing utilities and popular datasets for NLP
from torchtext import datasets
import random
import torch.nn as nn
import time
import math
import torch.optim as optim

# We fix the seeds to get consistent results for each training.

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Helper function to print time between epochs


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# %%
TEXT = data.Field(tokenize='spacy')

# We will be using the WikiText-2 corpus, which is a popular LM dataset.
# The WikiText language modeling dataset is a collection of texts extracted
# from good and featured articles on Wikipedia.
# It contains about 2 million words
train_data, valid_data, test_data = datasets.WikiText2.splits(TEXT)

# Data stats
print('train.fields', train_data.fields)
print('len(train)', len(train_data))

# Build a vocabulary out of tokens available from the pre-trained embeddings list and the vocabulary of labels
TEXT.build_vocab(train_data, vectors="glove.6B.100d")
print('Text Vocabulary Length', len(TEXT.vocab))

BATCH_SIZE = 64

# place the tensors on the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# %%

# BPTTIterator (Backpropagation Through Time Iterator)
# divides the corpus into batches of [sequence length, bptt_len]

train_iterator, valid_iterator, test_iterator = data.BPTTIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE, bptt_len=30,
    device=device, repeat=False)

for batch in train_iterator:
    demo_batch = batch
    break

print(demo_batch)

print()

# Note that the first dimension is the sequence, and the next is the batch.
# We can reshape this to [batch size, sentence length] as we did earlier with a transpose.
# The target is the original text offset by 1
print("Demo batch `text`:\n", demo_batch.text[:5, :3])
print("Demo batch `target`:\n", demo_batch.target[:5, :3])

# %%


class Manual_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.forget_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Sigmoid()
        )

        self.input_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Sigmoid()
        )

        self.candidate_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Tanh()
        )

        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x, prev_hidden):

        # shape(x) = [B, T, input_size]
        # shape(prev_hidden) = ([1, B, hidden_size], [1, B, hidden_size]) where 1 = num_layers * num_directions

        batch_size, sequence_length, _ = x.size()

        # At t=0, h_t and c_t will be initialized to a vector of 0s
        h_t = prev_hidden[0].squeeze(0)
        c_t = prev_hidden[1].squeeze(0)

        hidden_states = torch.zeros(
            batch_size, sequence_length, self.hidden_size).to(device)

        for t in range(sequence_length):

            # shape(x_t) = [B, input_size]
            x_t = x[:, t, :]

            # shape(concat_h_x) = [B, hidden_size+input_size]
            concat_h_x = torch.cat((h_t, x_t), dim=-1)

            # shape(f_t) = [B, hidden_size]
            f_t = self.forget_gate(concat_h_x)

            # shape(c_prime_t) = [B, hidden_size]
            c_prime_t = c_t * f_t

            # shape(i_t) = [B, hidden_size]
            # shape(cand_t) = [B, hidden_size]
            i_t = self.input_gate(concat_h_x)
            cand_t = self.candidate_gate(concat_h_x)

            # shape(c_t) = [B, hidden_size]
            c_t = c_prime_t + (i_t * cand_t)

            # shape(o_t) = [B, hidden_size]
            # shape(h_t) = [B, hidden_size]
            o_t = self.output_gate(concat_h_x)
            h_t = o_t * torch.tanh(c_t)

            hidden_states[:, t, :] = h_t

        h_t, c_t = h_t.unsqueeze(0), c_t.unsqueeze(0)
        return hidden_states, (h_t, c_t)

# %%


class RNN(nn.Module):

    # variant is a flag which is either: "rnn", "lstm", "manual_lstm"
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pad_idx, variant):

        super().__init__()

        self.variant = variant

        self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors)

        # UNIDIRECTIONAL RNN layer: For LM modelling we do not see/have access to the right context

        if variant == "rnn":
            self.rnn = nn.RNN(embedding_dim,
                              hidden_dim,
                              batch_first=True)
        elif variant == "lstm":
            self.rnn = nn.LSTM(embedding_dim,
                               hidden_dim,
                               batch_first=True)
        elif variant == "manual_lstm":
            self.rnn = Manual_LSTM(embedding_dim, hidden_dim)
        else:
            raise ValueError(
                "Expected `variant` to be one of 'rnn', 'lstm', or 'manual_lstm'")

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, prev_hidden):

        # shape(text) = [B, T]

        # If vanilla RNN:
            # shape(prev_hidden) = [1, B, D] where 1 = num_layers*num_directions
        # If LSTM:
            # prev_hidden is a tuple of previous hidden states and cell states: (ALL_HIDDEN_STATES, ALL_CELL_STATES)
            # shape(ALL_HIDDEN_STATES)=shape(ALL_CELL_STATES) = [1, B, D] where 1 = num_layers*num_directions

        embedded = self.dropout(self.embedding(text))
        # shape(embedded) = [B, T, E]

        all_hidden, last_hidden = self.rnn(embedded, prev_hidden)
        # shape(all_hidden) = [B, T, D]
        # shape(last_hidden) = [num layers, B, T]

        # Take all hidden states to produce an output word per time step
        logits = self.fc(self.dropout(all_hidden))
        # shape(logits) = [B, O]

        return logits, last_hidden

# %%


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(TEXT.vocab)
DROPOUT = 0.5
# get our pad token index from the vocabulary
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


# %%
def save_hidden(hidden):
    """Wraps hidden states in new Tensors, to declare it not to need gradients. So that the initial hidden state for this batch is constant and doesn’t depend on anything."""

    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(save_hidden(v) for v in hidden)


# %%
def perplexity(loss_per_token):
    return math.exp(loss_per_token)


# %%
def train(model, train_iterator, valid_iterator, optimizer, criterion, N_EPOCHS=25, is_lstm=False, force_stop=False):

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        model.train()

        epoch_loss = 0
        epoch_items = 0

        # The `1` is the number of layers * number of directions.
        # i.e. we have 1 layer and we are moving in 1 direction
        # More info: https://pytorch.org/docs/stable/nn.html#rnn
        prev_hidden = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM, device=device)
        if is_lstm:
            prev_ht = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM, device=device)
            prev_ct = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM, device=device)
            prev_hidden = (prev_ht, prev_ct)

        # `batch` is a tuple of Tensors: (TEXT, TARGET)
        for i, batch in enumerate(train_iterator):

            if force_stop:
                print("Currently processing train batch {} of {}".format(
                    i, len(train_iterator)))
                if i % 7 == 0 and i != 0:
                    break

            # Zero the gradients
            optimizer.zero_grad()

            text = batch.text
            targets = batch.target
            # shape(text) = [T, B]
            # shape(target) = [T, B]

            # We reshape text and target to [B, T].
            text = text.T
            targets = targets.T
            # shape(text) = [B, T]
            # shape(target) = [B, T]

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # Otherwise the model would backpropagate all the way to beginning of the dataset.
            prev_hidden = save_hidden(prev_hidden)
            logits, prev_hidden = model(text, prev_hidden)

            # Compute the loss
            # We reshape inputs to eliminate batching
            loss = criterion(logits.view(-1, OUTPUT_DIM), targets.reshape(-1))

            # backprop the average loss and update parameters
            # Why average loss?
            loss.mean().backward()

            # update the parameters using the gradients and optimizer algorithm
            optimizer.step()

            epoch_loss += loss.detach().sum()
            epoch_items += loss.numel()

        # We compute loss per token for an epoch
        train_loss_per_token = epoch_loss / epoch_items
        # We compute perplexity
        train_ppl = perplexity(train_loss_per_token)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        valid_loss_per_token, valid_ppl = evaluate(model,
                                                   valid_iterator,
                                                   criterion,
                                                   is_lstm=is_lstm,
                                                   force_stop=force_stop)

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss_per_token:.3f} | Train Perplexity: {train_ppl:.3f}')
        print(
            f'\t Val. Loss: {valid_loss_per_token:.3f} |  Val. Perplexity: {valid_ppl:.3f}')

        if force_stop:
            break

# %%


def evaluate(model, iterator, criterion, is_lstm=False, force_stop=False):

    model.eval()

    epoch_loss = 0
    epoch_items = 0

    # we initialise the first hidden state with zeros
    prev_hidden = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM, device=device)
    if is_lstm:
        prev_ht = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM, device=device)
        prev_ct = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM, device=device)
        prev_hidden = (prev_ht, prev_ct)

    with torch.no_grad():
        for i, batch in enumerate(iterator):

            if force_stop and i % 3 == 0 and i != 0:
                print("Currently processing valid batch {} of {}".format(
                    i, len(train_iterator)))
                break

            text, target = batch.text, batch.target
            text, target = text.T, target.T
            logits, prev_hidden = model(text, prev_hidden)

            # compute the loss
            loss = criterion(logits.view(-1, OUTPUT_DIM), target.reshape(-1))

            prev_hidden = save_hidden(prev_hidden)

            epoch_loss += loss.detach().sum()
            epoch_items += loss.numel()

        loss_per_token = epoch_loss / epoch_items
        ppl = math.exp(loss_per_token)

    return loss_per_token, ppl


# %%
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

# %%
os.remove(".vector_cache\glove.6B.50d.txt")
os.remove(".vector_cache\glove.6B.200d.txt")
os.remove(".vector_cache\glove.6B.300d.txt")
os.remove(".vector_cache\glove.6B.300d.txt.pt")

# %%

rnn_model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                DROPOUT,
                PAD_IDX,
                variant="rnn")

rnn_model = rnn_model.to(device)
optimizer = optim.Adam(rnn_model.parameters())
train(rnn_model, train_iterator, valid_iterator,
      optimizer, criterion, force_stop=False)
torch.save(rnn_model.state_dict(), "LM_models\rnn_lm.pt")


manual_lstm = RNN(INPUT_DIM,
                  EMBEDDING_DIM,
                  HIDDEN_DIM,
                  OUTPUT_DIM,
                  DROPOUT,
                  PAD_IDX,
                  variant="manual_lstm")

manual_lstm.to(device)
optimizer = optim.Adam(manual_lstm.parameters())
train(manual_lstm, train_iterator, valid_iterator,
      optimizer, criterion, is_lstm=True)
torch.save(manual_lstm.state_dict(), "LM_models\manual_lstm.pt")


lstm = RNN(INPUT_DIM,
           EMBEDDING_DIM,
           HIDDEN_DIM,
           OUTPUT_DIM,
           DROPOUT,
           PAD_IDX,
           variant="lstm")

lstm.to(device)
optimizer = optim.Adam(lstm.parameters())
train(lstm, train_iterator, valid_iterator, optimizer,
      criterion, is_lstm=True)
torch.save(lstm.state_dict(), "LM_models\lstm.pt")
