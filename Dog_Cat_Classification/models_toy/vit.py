# ViT核心在于预处理和后处理，中间的backbone和Transformer是一样的。这里给出实现大纲：
# - 预处理
#   - 划分16*16的patch
#   - Linear Projection映射为base token
#   - 为最终分类添加CLS token
#   - 添加position embedding
# - Transformer Backbone
# - MLP 基于CLS Token分类
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class P3(nn.Module):
    """
    patch position process stage
    """
    def __init__(self, image_size, patch_size, patch_dim, dim):
        super().__init__()
        # pdb.set_trace()
        self.patch_size = patch_size
        self.image_size = image_size
        self.patch_dim = patch_dim
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2
        self.linear_projection = nn.Linear(patch_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))# 基于位置的embedding，广播
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))   # 维度要和(B,L,C)对齐

    def forward(self, x):
        # x: (B, C, H, W)
        # patchify
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        # x:(batch_size, length(token_nums), token_dim)
        # linear projection
        patches = self.linear_projection(patches)
        # add cls token
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=x.size(0))
        # concat cls token
        patches = torch.cat((cls_token, patches), dim=1)
        # add position embedding
        patches += self.position_embedding
        return patches
    
class Multihead_self_attention(nn.Module):
    def __init__(self, heads, head_dim, dim) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.inner_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # q: (B, H, L, D), k: (B, H, L, D), v: (B, H, L, D)
        dots = torch.einsum('bhid, bhjd -> bhij', q, k) * self.scale
        attn = self.softmax(dots)
        out = torch.einsum('bhij, bhjd -> bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.norm(x + out)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.net(x))

class Transformer_block(nn.Module):
    def __init__(self, dim, heads, head_dim, hidden_dim) -> None:
        super().__init__()
        self.attention = Multihead_self_attention(heads, head_dim, dim)
        self.feed_forward = FeedForward(dim, hidden_dim)
    
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x
    
class ViT_Toy(nn.Module):
    def __init__(self, image_size, patch_dim, patch_size, dim, heads, mlp_dim, depth, num_classes) -> None:
        super().__init__()
        # pdb.set_trace()
        self.p3 = P3(image_size, patch_size, patch_dim, dim)
        self.transformer = nn.Sequential(*[Transformer_block(dim, heads, dim // heads, mlp_dim) for _ in range(depth)])
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.p3(x)
        x = self.transformer(x)
        cls_tokens = self.to_cls_token(x[:, 0])
        return self.mlp_head(cls_tokens)


if __name__ == '__main__':
    model = ViT_Toy(image_size=224, patch_size=16, patch_dim=768, dim=768, heads=12, mlp_dim=3072, depth=6, num_classes=2)
    # 假定的输入参数
    B = 4  # 批量大小
    C = 3  # 通道数
    H = W = 224  # 图像的高度和宽度

    # 创建一个随机图像批量作为输入
    inputs = torch.randn(B, C, H, W)
    outputs = model(inputs)
    print("output", outputs)
