# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_expands=4, dropout=0.1,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_expands, dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_expands, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model  # hidden_dim == channel
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        # c ==  hidden_dim, 和ViT类似，采用一个卷积核进行patch_embedding，将图像特征映射到hidden_dim
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        # src = rearrange(src, 'b c h w -> (h w) b c')
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # pos_embed = rearrange(pos_embed, 'b c h w -> (h w) b c')
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        #print(f"transformer -> {hs.shape}")
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            #print(f"decoder layer out -> {output.shape}")
            if self.return_intermediate and self.norm is not None:
                intermediate.append(self.norm(output))  # ！！可能有问题，确保self.norm不为空

        if self.norm is not None:
            output = self.norm(output)

            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_expaned=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.mlp = Mlp(in_features=d_model, hidden_features=d_model*dim_expaned,
                       out_features=d_model, drop_prob=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                src,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        # self.attn返回一个元组，第一个是注意力计算机的结果，即注意力权重的加权和Q，K和V相乘的结果。
        # 第二个元素是注意力的权重，即Q和K相乘的结果。
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = src + self.dropout2(self.mlp(src))
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_expaned=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.mlp = Mlp(in_features=d_model, hidden_features=d_model*dim_expaned,
                       out_features=d_model, drop_prob=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        tgt=object query: -> [num_query,batch,hidden_dim]
        hidden_dim就是从骨干网络经过projection后输入进来特征图的通道数；
        memory就是encoder部分的输出: -> [height x width, batch, hidden_dim]
        pos: -> 输入到encoder内的位置编码 [height x width, batch, hidden_dim]
        query_pos; -> tgt的位置编码 [num_query, batch, hidden_dim]
        memory_key_padding_mask: -> encoder和encoderlayer的src_key_padding_masks
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_key_padding_mask)[0]
        #print(f"first attn -> {tgt2.shape}")
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, key_padding_mask=memory_key_padding_mask)[0]
        #print(f"second attn -> {tgt2.shape}")
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt = tgt + self.dropout3(self.mlp(tgt))
        tgt = self.norm3(tgt)
        return tgt


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_prob=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.linear2(x)
        return x


def build_transformer(hidden_dim=256,
                      dropout=0.1,
                      nheads=8,
                      dim_expands=4,
                      enc_layers=6,
                      dec_layers=6):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_expands=dim_expands,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        return_intermediate_dec=True,
    )


