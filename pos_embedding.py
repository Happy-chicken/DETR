import torch
from torch import nn
import math
from einops import rearrange, repeat
from util.misc import NestedTensor


class PositonembeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 生成位置特征的数量
        self.temperature = temperature  # sin函数缩放的倍数
        self.normalize = normalize  # 是否归一化
        if scale is not None and normalize is False:  # 缩放范围不为空时，不许设置为True
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi  # 在2*pi内缩放
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):  # facebook的源码中重新写了一个tensor类，成员有tensor和mask
        """
        x: 输入
        mask: 掩码矩阵
        为了统一batch中的所有图像尺寸，先得到所有维度中最大的尺寸，再对较小的图像进行padding，
        mask矩阵就是设置那些区域是padding得到的，以便模型能够学到有用的知识。
        """
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask  # 得到的就是那些地方是真实有效的，是真正的图像部分。mask -> [b, h, w]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 在列上累加，该列有多少个像素有效
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 在行上累加，该行有多少个像素有效
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 在最后一维上进行升维，pos_x, pos_y -> [b, h, w, 1] -> [b, h, w, num_pos_feats]
        # broadcast mechanism
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # 正余弦编码，偶数位置sin编码，奇数位置cos编码
        # 对于每个位置x y，所在列对应的编码值排在前num_pos_feat维，而行所对应的编码值在后num_pos_feat上
        # 特征图上的各个位置(h * w)都对应到不同的维度，2 * num_pos_feats的编码
        # stack: -> [batch, height, (width, num_pos_feats / 2), 2]
        # flatten: -> [b, h, w, num_pos_feats]
        # cat: -> [b, h, w, num_pos_feats * 2]
        # permute: -> [b, num_pos_feats * 2, h, w]
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # -> [b, num_pos_feats, h, w]
        return pos

class PositonembeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)  # search table
        self.col_embed = nn.Embedding(50, num_pos_feats)  # [b, c, h, w]->[..., num_pos_feats]

        self.reset_parameters()

    def reset_parameters(self):  # 对权重进行均匀分布，暂时不清楚其原因
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        height, width = x.shape[-2:]

        temp_row = torch.arange(width, device=x.device)  # 每行中的位置
        temp_col = torch.arange(height, device=x.device)  # 每列中的位置

        x_embed = self.col_embed(temp_row)  # --> [width, num_pos_feats]
        y_embed = self.row_embed(temp_col)  # --> [height, num_pos_feats]

        # cat: [height, width, num_pos_feats * 2]
        # permute: [num_pos_feats * 2, height, width]
        # unsqeueeze(0): [_, num_pos_feats * 2, height, width]
        # repeat: [batch, num_pos_feats * 2, height, width]
        pos = torch.cat([
            # unsqueeze(0) -> [_, width, num_pos_feats]
            # repeat -> [height, width, num_pos_feats]
            x_embed.unsqueeze(0).repeat(height, 1, 1),
            # unsqueeze(1) -> [height, _, num_pos_feats]
            # repeat -> [height, width, num_pos_feats]
            y_embed.unsqueeze(1).repeat(1, width, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        '''
        使用einops实现与上面同样的效果
        x_embed = repeat(rearrange(x_embed, 'w n ->  1 w n'), 
                        'h w n -> (repeat h) w n', repeat=height)

        y_embed = repeat(rearrange(y_embed, 'h n ->  h 1 n'), 
                        'h w n -> h (repeat w) n', repeat=width)

        pos = repeat(rearrange(torch.cat([x_embed, y_embed], dim=-1), 'h w n -> 1 n h w'),
                    'b n h w -> (repeat b) n h w', repeat=x.shape[0])
        '''

        return pos


def build_position_embedding(hidden_dim, method="learned"):
    N_steps = hidden_dim // 2
    if method == "learned":
        position_embedding = PositonembeddingLearned(N_steps)
    elif method == "sine":
        position_embedding = PositonembeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"method don't support {method}")
    return position_embedding