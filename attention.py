# -*- coding: utf-8 -*-

# Install the transformers library if necessary
pip install transformers

"""# Intro: Transformers for Sequence Classification

Typical Transformers can be broken down into the following components:
Remark: Here, we will focus on the encoding of sentences for the purpose of sentiment classification, the decoder used in sequence2sequences Transformers has a very similar structure.


1.   Embedding: An embedding layer that transforms word tokens into vector representations.
2.  Encoder: The encoder block consists of several multi-headed attention blocks

    2.1.   Attention block 1

    2.2.   Attention block 2

    ...

    2.L.   Attention block L

3. Pooling: The final output of the encoder computes one vector representation for each token. To further summarize/pool this information for classification, the [CLS] token at sequence position 0 is typically selected.

4. Classification: A standard small MLP is used for classification and outputs probabilites for the most likely predicted class.

## Preparation: From sentences to tokens

First, we will use hugginface's tokenizer to go from words to indices in the vocabulary. In the following, we will focus on the distilBERT model:
"""

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sentence = "This was one of the best movies I have ever seen."
inputs = tokenizer(sentence, return_tensors = 'pt')
print(inputs)

# Now we download pretrained weights for the model.
! wget https://tubcloud.tu-berlin.de/s/GfEoq4r8Sgb2727/download/distilbert.pt

# This config contains the model parameters, it will be important to understand what representations
# have which dimensionality

from torch import nn
import torch
import math
import torch

class Config(object):
    def __init__(self):
        self.n_heads = 12
        self.n_layers = 6
        self.pad_token_id = 0
        self.dim = 768
        self.hidden_dim = 3072
        self.max_position_embeddings = 512
        self.vocab_size = 30522
        self.eps = 1e-12

        self.attention_head_size = int(self.dim / self.n_heads)
        self.all_head_size = self.n_heads * self.attention_head_size

        self.n_classes = 2
        self.device = 'cpu'

config = Config()

"""# 1. Embedding Layer 
Next, we will have to implement the Embedding layer.


1.   forward: compute the output embeddings from the input_embeds and position_embeds



"""

torch.manual_seed(0) # set seed for reproducible random initialization of weights

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=config.eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings)
        """

        # Embedding the input ids
        input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        seq_length = input_embeds.size(1)

        # Creating and embedding the position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        # Compute the output embeddings

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        return embeddings

embedding_layer = Embeddings(config)
# Test if your embedding layer computes an output
embeddings = embedding_layer(inputs['input_ids'])

"""# 2. Attention Block 
The next step is writing the attention block. It mainly consits of the self-attention function, a layer normalization followed by an additional projection layer.

"""

torch.manual_seed(0)

class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # self-attention components

        self.q_lin = nn.Linear(config.dim, config.all_head_size)
        self.k_lin = nn.Linear(config.dim, config.all_head_size)
        self.v_lin = nn.Linear(config.dim, config.all_head_size)

        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim, bias=True)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=config.eps)

        # feed-forward network
        self.lin1 = nn.Linear(in_features=config.dim, out_features=3072, bias=True)
        self.lin2 = nn.Linear(in_features=3072, out_features=config.dim, bias=True)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=config.eps)


    def forward(self, hidden_states):

        def shape(x):
            """ separate heads """
            return x.view(1, -1, 12, 64).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(1, -1, 12 * 64)

        bs=hidden_states.shape[0]
        n_nodes= hidden_states.shape[1]

        query=key=value=hidden_states
        q = self.q_lin(query)
        k = self.k_lin(key)
        v = self.v_lin(value)

        # Separating the heads
        q = shape(q)  # (bs, n_heads, q_length, dim_per_head)
        k = shape(k)  # (bs, n_heads, k_length, dim_per_head)
        v = shape(v)  # (bs, n_heads, k_length, dim_per_head)

        # Normalizing the query-tensor
        q = q / math.sqrt(q.shape[-1])

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(2, 3))

        # Transform the scores into probability distribution via softmax
        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)

        # Compute the weighted representation of the value-tensor (aka context)
        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)

        # Merging the heads again
        context = unshape(context)  # (bs, q_length, dim)

        # Additional projection of the context to get the output of the self-attention block
        sa_output = self.out_lin(context)
        sa_output = self.sa_layer_norm(sa_output + hidden_states)

        # Feed-forward network to compute the attention block output
        x = self.lin1(sa_output)
        x = nn.functional.gelu(x)
        ffn_output = self.lin2(x)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)

        return  ffn_output, weights

block = AttentionBlock(config)
# Test if your attention block computes an output
block_output = block(embeddings)

"""# 3. Building the model 

"""

torch.manual_seed(0)

class DistillBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers=config.n_layers

        # embedding
        self.embeddings = Embeddings(config)

        # encoder

        self.attention_layers = torch.nn.Sequential(*[AttentionBlock(config) for i in range(config.n_layers)])

        # classification
        self.pre_classifier =  nn.Linear(in_features=config.dim, out_features=config.dim, bias=True)
        self.classifier =  nn.Linear(in_features=config.dim, out_features=config.n_classes, bias=True)

        self.attention_probs = {i: [] for i in range(config.n_layers)}

    def forward(self, input_ids):
        """
        Parameters:
            input_ids (torch.Tensor): torch.tensor(bs, max_seq_length) The token ids to embed.
        Returns: torch.tensor(bs, n_classes) The computed logit scores for each class.
        """

        # Computing the embeddings
        hidden_states =  self.embeddings(input_ids=input_ids).to(self.config.device)

        # Iteratively going through the attention layers
        encoder_input = hidden_states

        for i,block in enumerate(self.attention_layers):

            output, attention_probs = block(encoder_input)

            self.attention_probs[i] = attention_probs
            encoder_input = output

        # Pooling by selection the [CLS] token
        pooled_output = output[:, 0]  # (bs, dim)

        # Classification

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)

        return logits

model = DistillBertAttention(config)
state_dict = torch.load('distilbert.pt')
_ = model.load_state_dict(state_dict)
_ = model.eval()

# Predict your output
logits = model(inputs['input_ids'])

"""# 4. Visualize the attention weights 

Let's now look at what tokens the model selects in its self-attention blocks.
"""

import numpy as np
import matplotlib.pyplot as plt

# YOUR CODE HERE #
subtokens = tokenizer.convert_ids_to_tokens(list(inputs['input_ids'].squeeze()))
A = np.array([model.attention_probs[i].detach().cpu().numpy().squeeze() for i in range(config.n_layers)])
A_avg = A.mean(1)

for i in range(config.n_layers):
    f, ax  = plt.subplots(1,1)
    h = ax.imshow(A_avg[i], cmap='Reds', vmin=0, vmax=1)
    ax.set_xticks(range(len(subtokens)))
    ax.set_xticklabels(subtokens, rotation=45)

    ax.set_yticks(range(len(subtokens)))
    ax.set_yticklabels(subtokens)

    ax.set_title('layer {}'.format(i))
    plt.colorbar(h)
    plt.show()
