import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder:
    def __init__(self, embed_size, n=10000):
        self.embed_size = embed_size
        self.n = n
        
    def positional_encoding(self, seq_len):

        n = torch.IntTensor([self.n])
        d = self.embed_size

        pos_enc = torch.zeros((seq_len, d))

        for j in range(seq_len):
            for i in range(int(d/2)):
                p = torch.pow(n, 2*i/d)
                
                pos_enc[j, 2*i] = torch.sin(j/p)
                pos_enc[j, 2*i+1] = torch.cos(j/p)

        return pos_enc.unsqueeze(0)
    
    def __call__(self, seq_len):
        return self.positional_encoding(seq_len)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * self.heads == self.embed_size, 'Embed size needs to be divisible by heads'
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        # Split into self.heads number of heads
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)
        
        # values shape after: (N, self.heads, values_len, self.head_dim)
        values = self.values(values).permute(0, 2, 1, 3) 
        
        # keys shape after: (N, self.heads, self.head_dim, keys_len)
        keys = self.keys(keys).permute(0, 2, 3, 1)
        
        # queries shape after: (N, self.heads, queries_len, self.head_dim)
        queries = self.queries(queries).permute(0, 2, 1, 3)
        
        # e shape after: (N, self.heads, queries_len, keys_len)
        e = (queries @ keys) / (self.embed_size**(1/2))
        
        if mask is not None:
            # if mask at same point is 0 - shutdown this point - set to -inf, in softmax it will be 0
            e = e.masked_fill(mask == 0, -1e20)
        
        attention = torch.softmax(e, dim=3)

        # out shape after: (N, self.heads, queries_len, self.head_dim)
        out = attention @ values

        out = out.reshape(N, queries_len, self.heads*self.head_dim)
        
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_size, forward_expansion, dropout):
        super(FeedForwardNetwork, self).__init__()
        
        self.fc_1 = nn.Linear(embed_size, embed_size*forward_expansion)
        self.fc_2 = nn.Linear(embed_size*forward_expansion, embed_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.gelu(self.fc_1(x))
        x = self.dropout(self.fc_2(x))
        
        return x


class Block(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(Block, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        
        self.norm_1 = nn.LayerNorm(embed_size)
        self.norm_2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = FeedForwardNetwork(embed_size, forward_expansion, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, values, keys, queries, mask):
        attention = self.attention(values, keys, queries, mask)
        
        x = self.dropout(self.norm_1(attention + queries))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm_2(forward + x))
        
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device
    ):

        super(Encoder, self).__init__()
        
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_encoding = PositionalEncoder(embed_size)
        
        self.layers = nn.ModuleList(
            [
                Block(
                    embed_size,
                    heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout
                )
                
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        seq_len = x.shape[1]

        out = self.dropout(
            (self.word_embedding(x) + self.position_encoding(seq_len).to(self.device))
        )

        # In the Encoder the query, key, value are all the same, in the decoder it will change. 
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = Block(
            embed_size, heads, forward_expansion, dropout
        )
        
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length,
        device
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_encoding = PositionalEncoder(embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        seq_len = x.shape[1]
        
        x = self.dropout(
            (self.word_embedding(x) + self.position_encoding(seq_len).to(self.device))
        )

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

    
class Model(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        heads=8,
        forward_expansion=4,
        dropout=0,
        max_length=100,
        device='cpu'
    ):

        super(Model, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device
        ).to(device)
        
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device
        ).to(device)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self._init_weights()
        
    def _init_weights(self):
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(mod.weight, mean=0.0, std=0.02)

                if isinstance(mod, nn.Linear) and mod.bias is not None:
                    torch.nn.init.zeros_(mod.bias)

            elif isinstance(mod, nn.LayerNorm):
                torch.nn.init.zeros_(mod.bias)
                torch.nn.init.ones_(mod.weight)
    
    def get_num_params(self):
        return sum(par.numel() for par in self.parameters())
    
    def make_src_mask(self, src):
        # mask only vocab['<pad>'], because it shouldn't have an effect on output result
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        # mask next words during decoding, decoder shouldn't have access to them 
        N, trg_len = trg.shape
        
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):

        src_mask = self.make_src_mask(src)

        trg_mask = self.make_trg_mask(trg)

        enc_out = self.encoder(src, src_mask)

        out = self.decoder(trg, enc_out, src_mask, trg_mask)

        return out