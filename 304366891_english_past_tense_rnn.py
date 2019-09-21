"""
.----------------------------------------------------------------------------.
| This is a PyTorch implementation of the Annotated Encoder Decoder, an RNN  |
| described in Bahdanau et al., built using the tutorial at                  |
| https://bastings.github.io/annotated_encoder_decoder/,                     |
| for exploring Halle's hypothesis regarding the encoding of phonemes as     |
| feature vectors by a learner of a natural language.                        |
|                                                                            |
| Tested on Windows 10 and Linux, with python 3.6+                           |
| Required libraries: { install using   python -m pip install <lib_name>  }  |
| - pytorch                                                                  |
| - numpy                                                                    |
| - matplotlib                                                               |
| - scipy                                                                    |
|                                                                            |   
| Toggle hypotheses in lines 318-321; toggle tests in lines 553-580          |
|                                                                            |
| Author: Michael Glukhman 304366891 (c) 2019                                |
'----------------------------------------------------------------------------'
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

import random, math, time, numpy as np
from nn_utils import *

# Preparing the datasets
# ----------------------

verbs = read_verbs()                                        # from nn_utils.py
IPA_glossary = read_ipa_transcripts()                       # from nn_utils.py
IPA_verbs = transcribe_verbs(verbs, IPA_glossary)           # from nn_utils.py
random.shuffle(IPA_verbs)
regular_verbs =   [x[:2] for x in IPA_verbs if x[2]==True]
irregular_verbs = [x[:2] for x in IPA_verbs if x[2]==False]
len_total = len(IPA_verbs)
len_regular = len(regular_verbs)
len_irregular = len(irregular_verbs)
print("Datasets: {0} regular verb pairs; {1} irregular verb pairs; {2} total\n".format(
    len_regular, len_irregular, len_total))

nonce_verbs = [('wʌg', 'wʌgd'),             # nonce verbs ending with a standard English phoneme
               ('smɜɹf', 'smɜɹft'),
               ('ʃɑɹn', 'ʃɑɹnd'),
               ('gɹOk', 'gɹOkt'),
               ('mʊf','mʊft'),
               ('ʃtut','ʃtutɪd'),
               ('sɪlflE','sɪlflEd'),        # <- in homage to Watership Down :)

               ('bɑx', 'bɑxt'),             # foreign voiceless: expected to add /t/
               ('mɪθ', 'mɪθt'),             # <- this one could actually be an English verb!
               ('klæɬ', 'klæɬt'),
               ('ɪfɹɑħ', 'ɪfɹɑħt'),
               ('ɬɔh', 'ɬɔht'),
               ('kʌpiɕ', 'kʌpiɕt'),

               ('blʌʙ', 'blʌʙd'),           # foreign voiced: expected to add /d/
               ('ʧEkɔfskij', 'ʧEkɔfskijd'),              
               ('ɹɪʃʌljœ', 'ɹɪʃʌljœd'),
               ('dbæʕ', 'dbæʕd'),
               ('buʋ', 'buʋd'),               
               
               ('flɜʈ', 'flɜʈɪd'),          # foreign coronal stops: expected to add /ɪd/
               ('gɑɖ', 'gɑɖɪd')]

# Divide the datasets into training and test sets at a ratio of 2:1
training_set = regular_verbs[:len_regular*2//3]
test_set = regular_verbs[len_regular*2//3:]
IR_training_set = irregular_verbs[:len_irregular*2//3]
IR_test_set = irregular_verbs[len_irregular*2//3:]

UNK_TOKEN = '?'  
PAD_TOKEN = '_'    
BOW_TOKEN = '$'
EOW_TOKEN = '#'
phonemes = [UNK_TOKEN, PAD_TOKEN, BOW_TOKEN, EOW_TOKEN] + phonemes
random.shuffle(phonemes)
phoneme_idx = {ph:i for i,ph in enumerate(phonemes)}
idx_phoneme = {i:ph for i,ph in enumerate(phonemes)}

# Used for embedding phonemes as feature-vectors (see nn_utils.py)
ph_weights = torch.FloatTensor([[
    sonority(ph), backness(ph), VOT(ph), rounded(ph), palatalized(ph),
    lateral(ph), nasal(ph), sibilant(ph), trilled(ph), diphthong(ph)
] for ph in phonemes])

MAX_WORD_LEN = max([len(v2) for v1, v2 in (regular_verbs+irregular_verbs)])

# Generate data batches from the datasets, for training and testing
def data_gen(dataset=training_set, randomize=True, batch_size=32, num_batches=100, length=MAX_WORD_LEN):
    for i in range(num_batches):
        src = []; trg = []

        for _ in range(batch_size):
            pair = random.choice(dataset) if randomize else dataset[i]
            v1 = pair[0]
            v1 = v1 + PAD_TOKEN * (MAX_WORD_LEN-len(v1))    # pad word
            v1 = [phoneme_idx[ph] for ph in v1]
            src.append(v1)

            v2 = pair[1]
            v2 = BOW_TOKEN + v2 + PAD_TOKEN * (MAX_WORD_LEN-len(v2))    # pad word
            v2 = [phoneme_idx[ph] for ph in v2]
            trg.append(v2)

        src = torch.LongTensor(src)
        trg = torch.LongTensor(trg)        
        src_lengths = [length] * batch_size
        trg_lengths = [length+1] * batch_size
        yield Batch((src, src_lengths), (trg, trg_lengths), pad_index=phoneme_idx[PAD_TOKEN])


# The Model
# ---------

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)
    
    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.trg_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

""" 
Our encoder is a bi-directional GRU.
The code below reads in a source word (a sequence of phoneme embeddings) and produces
the hidden states. It also returns a final vector, a summary of the complete word, 
by concatenating the first and the last hidden states (they have both seen the whole
word, each in a different direction). We will use the final vector to initialize the 
decoder.
"""
class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
                
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]
        return output, final

"""
The decoder is a conditional GRU. Rather than starting with an empty state like the 
encoder, its initial hidden state results from a projection of the encoder final vector.
"""
class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5, bridge=True):
        super(Decoder, self).__init__()        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout                 
        self.rnn = nn.GRU(emb_size + 2*hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    
        # to initialize from the final encoder state
        self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size, hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, _ = self.attention(query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas    


# Putting it all together
""" Toggle lines 318-319 or 320-321 below for checking a different hypothesis """

def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    model = Autoencoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        # nn.Embedding.from_pretrained(ph_weights),       # Halle: feature vectors
        # nn.Embedding.from_pretrained(ph_weights),       # Halle: feature vectors
        nn.Embedding(src_vocab, emb_size),            # Null hypothesis: segments
        nn.Embedding(tgt_vocab, emb_size),            # Null hypothesis: segments
        Generator(hidden_size, tgt_vocab))
    return model


# Training
# --------

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    (masking and padding are tools for creating, and later interpreting, an equal length batch of data)
    """
    def __init__(self, src, trg, pad_index=0):
        
        src, src_lengths = src
        
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

# the training loop (1 epoch = 1 pass through the training data)

def run_epoch(data_iter, model, loss_compute, print_every=10):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(data_iter, 1):
        
        _, _, pre_output = model.forward(batch.src, batch.trg,
                                         batch.src_mask, batch.trg_mask,
                                         batch.src_lengths, batch.trg_lengths)
        loss = loss_compute(pre_output, batch.trg_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        
        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.nseqs, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))

# compute loss

class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm

# Print examples
# --------------

def greedy_decode(model, src, src_mask, src_lengths, max_len=100, sos_index=1, eos_index=None):
    """Greedily decode a word."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for _ in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, src_mask,
              prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_symbol = torch.max(prob, dim=1)
        next_symbol = next_symbol.data.item()
        output.append(next_symbol)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_symbol)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)
  
def lookup_words(x, vocab=None):
    x = [idx_phoneme[int(i)] for i in x]
    return ''.join(x).strip(PAD_TOKEN)

def print_examples(example_iter, model, n=3, max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   trg_eos_index=None, 
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    correct_count = 0
    correct_suffix_count = 0
    print()
    
    src_eos_index = trg_eos_index = phoneme_idx[EOW_TOKEN]
    trg_sos_index = phoneme_idx[BOW_TOKEN]
        
    for batch in example_iter:
      
        src = batch.src.cpu().numpy()[0, :]
        trg = batch.trg_y.cpu().numpy()[0, :]

        # remove End-of-Word token (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        trg = trg[:-1] if trg[-1] == trg_eos_index else trg      
      
        result, _ = greedy_decode(
          model, batch.src, batch.src_mask, batch.src_lengths,
          max_len=max_len, sos_index=trg_sos_index, eos_index=trg_eos_index)
        
        suffix_len = len(lookup_words(trg)) - len(lookup_words(src))
        correct = (lookup_words(trg) == lookup_words(result))
        correct_suffix = (lookup_words(trg)[-suffix_len:] == lookup_words(result)[-suffix_len:])

        print("Src: {0: <{w}}Trg: {1: <{w}}Pred: {2: <{w}}{3}{4}".format(
            lookup_words(src), lookup_words(trg), lookup_words(result),
            '' if correct else '*',
            '' if correct_suffix else '!', w=MAX_WORD_LEN+1))
        count += 1        
        if correct: correct_count += 1                  # count examples with fully-correct inflection
        if correct_suffix: correct_suffix_count += 1    # count examples with correct suffix usage    

        if count == n:
            print()
            break
    
    return correct_count, correct_suffix_count

# All together: training
# ----------------------

def train(model, dataset=training_set, num_epochs=10, lr=0.0003, num_batches=100, print_every=100):
    """Train an NMT model"""  
    criterion = nn.NLLLoss(reduction="mean", ignore_index=0)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    eval_data = list(data_gen(dataset=dataset, batch_size=1, num_batches=num_batches)) 
    dev_perplexities = []

    for epoch in range(num_epochs):
        
        print("Epoch %d" % epoch)
        # train
        model.train()
        data = data_gen(dataset=dataset, batch_size=32, num_batches=num_batches)
        run_epoch(data,
                  model,
                  SimpleLossCompute(model.generator, criterion, optim),
                  print_every=print_every)

        # evaluate
        model.eval()
        with torch.no_grad():
            print_examples(eval_data, model, n=3, max_len=MAX_WORD_LEN)
            perplexity = run_epoch(eval_data,
                                   model,
                                   SimpleLossCompute(model.generator, criterion, None))
            print("Validation perplexity: %f" % perplexity)
            dev_perplexities.append(perplexity)            
        
    return dev_perplexities

# All together: testing
# ---------------------

def test(model, dataset=test_set, n=1000):
    if n>len(dataset):
        n = len(dataset)    
    test_data = list(data_gen(dataset=dataset, randomize=False, batch_size=1, num_batches=n)) 
    n_correct, n_correct_suffix = print_examples(test_data, model, n=n, max_len=MAX_WORD_LEN)
    print("Fully-correct V2 form: {0} out of {1} ({2:.2f}%)".format(n_correct, n, n_correct*100.0/n))
    print("Inflected using correct suffix: {0} out of {1} ({2:.2f}%)".format(n_correct_suffix, n, n_correct_suffix*100.0/n))

num_symbols = max(idx_phoneme)+1
model = make_model(num_symbols, num_symbols, emb_size=10, hidden_size=256, num_layers=1, dropout=0)

""" Toggle the following paragraphs for the different experiment stages.
    Each time, uncomment only ONE so that the experimnt remain unbiased by previously-trained weights.
"""
''' Train and test with regular verbs only '''

print("\n~~~ Training on regular English verbs ~~~\n")
train(model, num_epochs=15, num_batches=100, print_every=20)
print("\n~~~ Testing on previously-unseen regular English verbs ~~~\n")
test(model)

''' Train and test with IRregular verbs only '''

# print("\n~~~ Training on IRregular English verbs ~~~\n")
# train(model, dataset=IR_training_set, num_epochs=15, num_batches=100, print_every=20)
# print("\n~~~ Testing on previously-unseen irregular English verbs ~~~\n")
# test(model, dataset=IR_test_set)

''' Train with both regular and irregular verbs, test with regular only '''

# full_training_set = training_set + IR_training_set
# print("\n~~~ Training on both regular AND irregular English verbs ~~~\n")
# train(model, dataset=full_training_set, num_epochs=15, num_batches=100, print_every=20)
# print("\n~~~ Testing on previously-unseen regular English verbs ~~~\n")
# test(model)

''' Train with the full regular verbs set, test on nonce verbs '''

# print("\n~~~ Training on the full set of regular English verbs ~~~\n")
# train(model, dataset=regular_verbs, num_epochs=15, num_batches=100, print_every=20)
# print("\n~~~ Testing on nonce verbs with alien (for English speakers) phonology ~~~\n")
# test(model, dataset=nonce_verbs)

# Plot the embeddings for English phonemes
# ----------------------------------------

# embed_weights = F.log_softmax(model.trg_embed.weight.detach()).numpy()
# print(embed_weights.shape)

# english_IPA = ['p','b','t','d','k','g','ʧ','ʤ','f','v','θ','ð','s','z','ʃ','ʒ','h',
#                'm','n','ŋ','ɹ','l','j','w','i','u','ɪ','ʊ','E','O','ɛ','ɜ','ʌ','Ø',
#                'ɔ','æ','Y','W','ɑ']
# embed_relevant = []; labels = []
# for ph,i in phoneme_idx.items():
#     if ph in english_IPA:
#         embed_relevant.append(embed_weights[i])
#         labels.append(ph+' ')

# linked = linkage(embed_relevant, 'single')
# plt.figure(figsize=(10, 7))
# dendrogram(linked,
#             orientation='top',
#             labels=labels,
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.show()