import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(0, d_model, 2)[np.newaxis, :],
        d_model,
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


class Dual_SelfAttention(nn.Module):
    def __init__(self, input_size, embed_size, heads):
        super(Dual_SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.input_size = input_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.head_input_size = input_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.ModuleList([nn.Linear(self.head_input_size, self.head_dim, bias=False) for _ in range(heads)])
        self.keys = nn.ModuleList([nn.Linear(self.head_input_size, self.head_dim, bias=False) for _ in range(heads)])
        self.queries = nn.ModuleList([nn.Linear(self.head_input_size, self.head_dim, bias=False) for _ in range(heads)])
        self.queries_u = nn.ModuleList([nn.Linear(self.head_input_size, self.head_dim, bias=False) for _ in range(heads)])

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size)

        self.W_d = nn.Parameter(torch.randn(self.head_dim, 2 * self.head_dim))
        self.h = nn.Parameter(torch.randn(self.head_dim, 1))
        nn.init.xavier_uniform_(self.W_d)
        nn.init.xavier_uniform_(self.h)

        self.positional_encoding = positional_encoding(1000, input_size * 2)  # Assuming a maximum sequence length of 1000

    def forward(self, inputs, inputs2, mask=None):                  #inputs is Y, inputs2 is Y_u
        values = inputs, keys = inputs, queries = inputs
        queries_u = inputs2
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        print(self.positional_encoding.shape)
        # Adding positional encoding
        values = values + self.positional_encoding[:, :value_len, :]
        keys = keys + self.positional_encoding[:, :key_len, :]
        queries = queries + self.positional_encoding[:, :query_len, :]
        queries_u = queries_u + self.positional_encoding[:, :query_len, :]

        values = values.reshape(N, value_len, self.heads, self.head_input_size)
        keys = keys.reshape(N, key_len, self.heads, self.head_input_size)
        queries = queries.reshape(N, query_len, self.heads, self.head_input_size)
        queries_u = queries_u.reshape(N, query_len, self.heads, self.head_input_size)

        energy1 = 0
        energy2 = 0

        head_outputs = []
        last_attention = []

        for i in range(self.heads):
            energy1 += torch.einsum("nqhd,nkhd->nhqk", [self.queries[i](queries[:, :, i, :]).reshape(N, query_len, 1, -1), self.keys[i](keys[:, :, i, :]).reshape(N, key_len, 1, -1)])
            energy2 += torch.einsum("nqhd,nkhd->nhqk", [self.queries_u[i](queries_u[:, :, i, :]).reshape(N, query_len, 1, -1), self.keys[i](keys[:, :, i, :]).reshape(N, key_len, 1, -1)])

            energy1 /= self.heads
            energy2 /= self.heads

            if mask is None:
                # Create the lower triangular mask to mask out positions j > i
                mask = torch.tril(torch.ones((query_len, key_len))).to(values.device)

            energy1 = energy1.masked_fill(mask == 0, float("-1e20"))
            energy2 = energy2.masked_fill(mask == 0, float("-1e20"))

            attention1 = torch.nn.functional.softmax(energy1, dim=3)
            attention2 = torch.nn.functional.softmax(energy2, dim=3)

            out1 = torch.einsum("nhql,nlhd->nhqd", [attention1, self.values[i](values[:, :, i, :]).reshape(N, value_len, 1, -1)]).reshape(N, query_len, self.head_dim)
            out2 = torch.einsum("nhql,nlhd->nhqd", [attention2, self.values[i](values[:, :, i, :]).reshape(N, value_len, 1, -1)]).reshape(N, query_len, self.head_dim)

            #attention mechanism to combine the two outputs
            concat1 = torch.cat((out1, inputs), dim=2)
            concat2 = torch.cat((out2, inputs), dim=2)

            E_c = torch.matmul(F.relu(torch.matmul(concat1, self.W_d.T)), self.h)
            E_u = torch.matmul(F.relu(torch.matmul(concat1, self.W_d.T)), self.h)

            #Compute attention coefficient A_i for all timesteps T
            numerator = torch.exp(E_c)
            denominator = torch.exp(E_c) + torch.exp(E_u)

            A = numerator / denominator       #dim(A) = (N, query_len, 1)
            A.expand_as(out1)
            out = A * out1 + (1 - A) * out2

            head_outputs.append(out)

        out = torch.cat(head_outputs, dim=2)
        out.reshape(N, query_len, -1)

        out = out + inputs
        out = self.layer_norm(out)

        return self.fc_out(out)

'''
# Assuming that your input sequence has an embedding size of 256 and you are using 8 attention heads
model = SelfAttention(input_size=128, embed_size=256, heads=8)

# Define your input sequence (batch size=32, sequence length=10, embedding size=256)
input_sequence = torch.randn(32, 10, 128)

# Generate a mask (if necessary)
mask = None

# Forward pass through the model
output, attention_weights = model(input_sequence, input_sequence, input_sequence, mask)
print(output.shape)
print(attention_weights.shape)
'''