import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer()   # 相当于nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def window_partition(x, window_size):
    # x是特征图 (BS, H, W, C)
    # window_size是窗口大小 int
    BS, H, W, C = x.shape
    x = x.view(BS, H//window_size, window_size, W//window_size, window_size, C)  # (BS, H//window_size, window_size, W//window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # x: (BS, H//window_size, W//window_size, window_size, window_size, C) -> (BS*num_windows, window_size, window_size, C)
    return x   # (BS*num_windows, window_size, window_size, C)



def window_reverse(x, window_size, H, W):
    # x是需要还原的特征图 (BS*num_windows, window_size, window_size, C)
    # window_size是窗口大小
    # H,W 是原特征图的高和宽
    num_windows = (H//window_size) * (W//window_size)
    BS = x.shape[0] // num_windows
    x = x.view(BS, H//window_size, W//window_size, window_size, window_size, -1)  # (BS, H//window_size, W//window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(BS, H, W, -1)    # (BS, H//window_size, window_size, W//window_size, window_size, C)->(BS, H, W, C)
    return x   # (BS, H, W, C)




class WindowAttention(nn.Module):
    def __init__(self, embed_size, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0):
        """
        embed_size: 窗口中每个token的嵌入维度
        window_size: 窗口的高和宽 tuple (Wh, Ww) 一般Wh=Ww，可以计算总的token数
        num_heads: 多头注意力的头数
        """
        super(WindowAttention, self).__init__()
        self.embed_size = embed_size
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale or (self.head_dim)**(-0.5)    # QK^T/sqrt(dk)

        # 构造相对位置偏置表relative_position_bias_table
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*self.window_size[0]-1)*(2*self.window_size[1]-1), self.num_heads))  # ((2*Wh-1) * (2*Ww-1), num_heads)
        # 窗口大小为(Wh, Ww)，则相对行位置索引取值区间为[-(Wh-1),(Wh-1)]，共2*(Wh-1)+1=2*Wh-1个取值，相对列位置索引取值区间为[-(Ww-1),(Ww-1)]，共2*(Ww-1)+1=2*Ww-1个取值
        # 相对行位置索引和相对列位置索引进行组合共有(2*Wh-1) * (2*Ww-1)种相对位置组合，所以相对位置偏置表大小也为(2*Wh-1) * (2*Ww-1)
        # 因为有num_heads个头的注意力，所以相对位置偏置表大小为((2*Wh-1) * (2*Ww-1), num_heads)
        # nn.Parameter可以将这个张量注册为模型的可学习参数，这意味着这个张量会在训练过程中被不断优化（将不可训练的张量转换为可训练的参数，并将这个参数绑定到模型中）


        # 为窗口中的每个token构造相对位置索引，这里假设window_size=(7, 7) 即Wh=Ww=7
        coords_h = torch.arange(self.window_size[0])  # (0, 1, 2, 3, ..., 6)
        coords_w = torch.arange(self.window_size[1])  # (0, 1, 2, 3, ..., 6)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, 7, 7)
        # torch.meshgrid返回两个张量，分别是行索引和列索引，形状均为(Wh, Ww)=(7, 7)，通过torch.stack进行堆叠后形状为(2, Wh, Ww)=(2, 7, 7)
        # 其中coords[0,:,:]表示窗口中所有token的行索引，coords[1,:,:]表示窗口中所有token的列索引

        coords_flatten = coords.flatten(1)  # (2, Wh*Ww)=(2, 49),49表示窗口中49个token的行索引或列索引
        # coords_flatten[0, :]表示窗口中49个token的行索引[0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,...,6,6,6,6,6,6,6]
        # coords_flatten[1, :]表示窗口中49个token的列索引[0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,5,6,...,0,1,2,3,4,5,6]

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, 1)-(2, 1, Wh*Ww)=(2, Wh*Ww, Wh*Ww)=(2, 49, 49)
        # 其中relative_coords[0, :, :]表示窗口内49个token之间的相对行索引 (Wh*Ww, Wh*Ww)=(49, 49)
        # 其中relative_coords[1, :, :]表示窗口内49个token之间的相对列索引 (Wh*Ww, Wh*Ww)=(49, 49)

        relative_row_index = relative_coords[0, :, :]  # 相对行索引 (Wh*Ww, Wh*Ww)=(49, 49)
        relative_col_index = relative_coords[1, :, :]  # 相对列索引 (Wh*Ww, Wh*Ww)=(49, 49)

        # 下面将二元相对位置索引转换为一元相对位置索引
        # 将相对位置索引进行移位(+(Wh-1), +(Ww-1))，使得所有相对位置索引从0开始   (为什么加(Wh-1)或(Ww-1)，因为最小的相对位置索引为-(Wh-1)或-(Ww-1))
        relative_row_index += (self.window_size[0] - 1)  # (49, 49)
        relative_col_index += (self.window_size[1] - 1)  # (49, 49)

        # 然后将所有的相对行索引乘(2*Wh-1)
        relative_row_index *= (2 * self.window_size[0] - 1)  # (49, 49)

        # 最后将相对行索引和相对列索引相加得到一元的相对位置索引
        relative_position_index = relative_row_index + relative_col_index  # (49, 49) 相对位置索引的取值从0到(2*Wh-1)(2*Ww-1)-1，相对位置索引共有(2*Wh-1)(2*Ww-1)

        # 最后将相对位置索引relative_position_index注册为模型的一部分，但不会视为模型的参数
        self.register_buffer("relative_position_index", relative_position_index)  # (49, 49)
        # self.register_buffer: 会将张量注册为模型的一部分，但不会视为模型的参数，不会被优化器优化更新
        # 使用self.register_buffer的一个常用场景是存储一些固定的数据，这些数据需要在训练和推理过程中保持一致，并随着模型状态一同保存和加载，例如可以用来存储位置编码，固定的掩码或某些固定的统计值

        self.qkv = nn.Linear(embed_size, 3 * embed_size, bias=self.qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(embed_size, embed_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # 输入x: (BS*num_windows, N=window_size*window_size, embed_size)
        # mask: (num_windows, N, N)  N=window_size*window_size为总的tokens数
        B_, N, embed_size = x.shape   # B_为BS*num_windows，N为窗口中总的tokens数=window_size*window_size，embed_size为每个token嵌入的维度
        qkv = self.qkv(x)  # (BS*num_windows, N, embed_size)->(BS*num_windows, N, embed_size*3)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, BS*num_windows, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (BS*num_windows, num_heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.qk_scale   # (BS*num_windows, num_heads, N, N)

        attn = attn + self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1).permute(2, 0, 1).contiguous().unsqueeze(0)  # 使用相对位置索引从相对位置偏置表中将相对位置偏置B提取出来加到attn上
        # attn:(BS*num_windows, num_heads=8, N=49, N=49)
        # self.relative_position_bias_table(self.relative_position_index.view(-1)).view(N, N, -1).permute(2, 0, 1).contiguous().unsqueeze(0)
        # (169, 8) (49*49)->(49*49, 8)->(49, 49, 8)->(8, 49, 49)->(1, 8, 49, 49)=(1, num_heads, N, N)


        if mask is not None:  # 移动窗口注意力需要mask，mask不为None，mask: (num_windows, N, N)  其中num_windows=8*8=64, N=num_tokens=window_size*window_size
            num_windows = mask.shape[0]
            attn = attn.view(B_//num_windows, num_windows, self.num_heads, N, N) + mask.unsqueeze(0).unsqueeze(2)
            # (BS, num_windows, num_heads, N, N) + (1, num_windows, 1, N, N) -> (BS, num_windows, num_heads, N, N)
            # mask (N, N) 中的元素可取-100或0，-100表示掩码注意力（说明这是两个不同掩码区域的token之间的相似度），0表示不掩码注意力（说明这是两个相同掩码区域的token之间的相似度）
            attn = attn.view(-1, self.num_heads, N, N)  # (B_=BS*num_heads, num_heads, N, N)  恢复attn原本的形状
            attn = self.softmax(attn)
        else:  # 普通的窗口注意力不需要mask，mask为None
            attn = self.softmax(attn)


        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, embed_size)         # (BS*num_windows, num_heads, N, head_dim)->(BS*num_windows, N, num_heads, head_dim)
        x = self.proj(x)    # (BS*num_windows, N, embed_size)
        x = self.proj_drop(x)
        return x    # (BS*num_windows, N, embed_size)







class SwinTransformerBlock(nn.Module):
    def __init__(
            self,
            embed_size,               # 每个token的嵌入维度，也就是输入特征图的通道数
            input_resolution,         # 输入特征图的高宽(H, W)
            num_heads,                # 多头注意力的头数
            window_size=7,            # 窗口的大小，默认窗口中包含7*7个tokens
            shift_size=0,             # 移位的大小，一般为窗口大小的一半即window_size//2=3
            mlp_ratio=4,              # MLP中隐藏层相对于embed_size的倍数
            qkv_bias=True,            # qkv的线性变换中的偏置
            qk_scale=None,            # 对Q(K^T)进行缩放的量，其实就是1/sqrt(head_dim)
            drop=0,                   # 模块中使用的dropout
            attn_drop=0,              # 注意力中使用的dropout
            drop_path=0,              #
            act_layer=nn.GELU,        # MLP中使用的激活函数
            norm_layer=nn.LayerNorm,  # 模块中使用的层归一化函数
            fused_window_process=False,   #
    ):
        super(SwinTransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) < self.window_size:   # 窗口大小比输入特征图最小边还要大，就不要移位，并且将窗口大小置为最小边长度
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"


        self.norm1 = norm_layer(embed_size)     # 其实相当于nn.LayerNorm(embed_size)
        self.window_attention = WindowAttention(embed_size=embed_size,
                                                window_size=(window_size, window_size),
                                                num_heads=num_heads,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                attn_drop=attn_drop,
                                                proj_drop=drop)

        self.norm2 = norm_layer(embed_size)   # 两个LayerNorm是不一样的，所以需要定义两个
        self.mlp = Mlp(in_features=embed_size, hidden_features=int(embed_size*mlp_ratio), act_layer=act_layer, drop=drop)

        if self.shift_size > 0:   # 需要移动窗口，则attn_mask不为None，需要计算attn_mask
            H, W = self.input_resolution  # 输入特征图的高宽
            mask = torch.zeros(1, H, W, 1)  # 输入特征图形状为(BS, H, W, 1)，与其对应的mask形状为(1, H, W, 1)  第一个1表示BS维度，最后一个1表示通道维度

            # 用slice来对mask进行切片，切片获得的区域如图所示，这样获得的mask可以防止循环位移后不相邻token之间在窗口内计算注意力
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))

            # 根据上面的切片对mask进行区域划分
            cnt = 0  # 开始的区域设为0，区域从0开始逐渐变为1,2,3,4,5,6,7,8
            for h in h_slices:
                for w in w_slices:
                    mask[:, h, w, :] = cnt
                    cnt += 1

            # 划分区域后的mask形状仍为(1, H, W, 1)
            # 计算注意力是在窗口中进行的，因此也需要对mask进行窗口划分，窗口注意力的形状为(BS, num_windows, num_heads, N, N)，其中N为窗口中总的tokens数=window_size*window_size=7*7=49
            mask_windows = window_partition(mask, self.window_size)
            # mask_windows: (BS*num_windows, window_size, window_size, C)=(1*H//window_size*W//window_size, window_size, window_size, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)   # (num_windows, N)    N=window_size*window_size
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # (num_windows, 1, N)-(num_windows, N, 1)=(num_windows, N, N)
            # 相减的过程中，同一区域的两个token的attn_mask为0，不同区域的两个token的attn_mask不为0，这样可以根据attn_mask中的数是否为0判断是否要掩码对应两个token之间的注意力
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100)).masked_fill(attn_mask == 0, float(0))   # (num_windows, N, N)

        else:   # 不需要移动窗口，则attn_mask为None
            attn_mask = None


        self.register_buffer("attn_mask", attn_mask)
        # 这样可以在这个类中使用self.attn_mask来获取attn_mask这个张量
        # 这个参数可以像模型参数一样被保存，但不会被优化器更新



    def forward(self, x):   # (BS, L=H*W, embed_size)
        H, W = self.input_resolution  # 输入特征图的高宽
        BS, L, embed_size = x.shape   # BS为batch_size, L=H*W, embed_size为嵌入维度也是输入特征图的通道数

        assert L == H * W, "input feature has wrong size"
        shortcut = x  # (BS, L=H*W, embed_size)
        x = self.norm1(x)  # (BS, L=H*W, embed_size)
        x = x.view(BS, H, W, embed_size)   # (BS, H, W, embed_size)

        # 循环移位
        if self.shift_size > 0:  # 窗口需要移动
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))   # 在H和W维度进行移位  (BS, H, W, embed_size)
            shifted_x_windows = window_partition(shifted_x, self.window_size)  # 对移位后的特征图进行窗口划分   (BS*num_windows, window_size, window_size, embed_size)
            # (BS, H, W, embed_size)->(BS * H//window_size * W//window_size, window_size, window_size, embed_size)=(BS*num_windows, window_size, window_size, embed_size)
        else:
            shifted_x = x
            shifted_x_windows = window_partition(shifted_x, self.window_size)  # (BS*num_windows, window_size, window_size, embed_size)

        shifted_x_windows = shifted_x_windows.view(-1, self.window_size * self.window_size, embed_size)   # (BS*num_windows, N=window_size*window_size, embed_size)
        shifted_x_windows = self.window_attention(shifted_x_windows, mask=self.attn_mask)  # (BS*num_windows, N=window_size*window_size, embed_size)  (num_windows, N, N)
        # 获得的x: # (BS*num_windows, N, embed_size)  这里经过W-MSA/SW-MSA得到的结果是移位后获得的，需要将其逆移位还原回去

        shifted_x_windows = shifted_x_windows.view(-1, self.window_size, self.window_size, embed_size)   # (BS*num_windows, window_size, window_size, embed_size)
        shifted_x = window_reverse(shifted_x_windows, self.window_size, H, W)  # (BS, H, W, embed_size)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))  # (BS, H, W, embed_size)
        else:
            x = shifted_x  # (BS, H, W, embed_size)

        x = x.view(BS, L, embed_size)  # 将经过W-MSA/SW-MSA的x形状变为(BS, L=H*W, embed_size)  方便与shortcut进行相加
        x = x + shortcut   # (BS, L=H*W, embed_size)

        x = self.mlp(self.norm2(x)) + x  # (BS, L=H*W, embed_size)
        return x   # (BS, L=H*W, embed_size)



class PatchMerging(nn.Module):
    def __init__(self, input_resolution, embed_size, norm_layer=nn.LayerNorm):
        """
        input_resolution: 输入特征图的高和宽 (H, W)
        embed_size: 输入特征图的通道数，也是每个token的嵌入维度
        norm_layer: 要用到LayerNorm
        """
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.embed_size = embed_size
        self.norm = norm_layer(4 * embed_size)  # 相当于nn.LayerNorm(4 * embed_size)
        self.reduction = nn.Linear(4 * embed_size, 2 * embed_size, bias=False)

    def forward(self, x):  # x:(BS, L=H*W, embed_size)
        H, W = self.input_resolution
        BS, L, embed_size = x.shape

        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H} and {W}) must be even)"
        x = x.view(BS, H, W, embed_size)  # (BS, H, W, embed_size)
        x0 = x[:, 0::2, 0::2, :]  # (BS, H//2, W//2, embed_size)
        x1 = x[:, 0::2, 1::2, :]  # (BS, H//2, W//2, embed_size)
        x2 = x[:, 1::2, 0::2, :]  # (BS, H//2, W//2, embed_size)
        x3 = x[:, 1::2, 1::2, :]  # (BS, H//2, W//2, embed_size)
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (BS, H//2, W//2, 4*embed_size)
        x = x.view(BS, -1, 4 * embed_size)  # (BS, H//2*W//2, 4*embed_size)
        x = self.norm(x)  # (BS, H//2*W//2, 4*embed_size)
        x = self.reduction(x)   # (BS, H//2*W//2, 2*embed_size)
        return x   # (BS, L' = H//2*W//2, 2*embed_size)




class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage"""
    def __init__(
            self,
            embed_size,  # 每个token的嵌入维度，也就是输入特征图的通道数
            input_resolution,  # 输入特征图的高宽(H, W)
            num_heads,  # 多头注意力的头数
            depth,   # 表示这个layer中TransformerBlock的个数
            window_size=7,  # 窗口的大小，默认窗口中包含7*7个tokens
            # shift_size=0,  # 移位的大小，一般为窗口大小的一半即window_size//2=3
            # mlp_ratio=4,  # MLP中隐藏层相对于embed_size的倍数
            # qkv_bias=True,  # qkv的线性变换中的偏置
            # qk_scale=None,  # 对Q(K^T)进行缩放的量，其实就是1/sqrt(head_dim)
            # drop=0,  # 模块中使用的dropout
            # attn_drop=0,  # 注意力中使用的dropout
            # drop_path=0,  #
            # act_layer=nn.GELU,  # MLP中使用的激活函数
            # norm_layer=nn.LayerNorm,  # 模块中使用的层归一化函数
            # fused_window_process=False,  #
            downsample=None,
    ):
        super(BasicLayer, self).__init__()
        self.embed_size = embed_size
        self.input_resolution = input_resolution
        self.depth = depth

        # SwinTransformerBLocks
        self.blocks = nn.ModuleList(
            [SwinTransformerBlock(
                embed_size=embed_size,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size//2) for i in range(depth)]
        )

        # PatchMerging Layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, embed_size)
        else:
            self.downsample = None



    def forward(self, x):   # x:(BS, L=H*W, embed_size)
        for block in self.blocks:
            x = block(x)     # (BS, L=H*W, embed_size)

        if self.downsample is not None:
            x = self.downsample(x)  # (BS, L'=H//2*W//2, 2*embed_size)

        return x  # (BS, L'=H//2*W//2, 2*embed_size) or (BS, L=H*W, embed_size)




class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size. Default: 224.
        patch_size (int): Patch size. Default: 4.
        in_channels (int): Number of input image channels. Default: 3
        embed_size (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_size=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size)  # 图片大小，默认为(img_size, img_size)=(224, 224)
        self.patch_size = (patch_size, patch_size)  # patch大小，默认为(patch_size, patch_size)=(4, 4)
        self.patches_resolution = [self.img_size[0]//self.patch_size[0], self.img_size[1]//self.patch_size[1]]  # (H//Ph, W//Pw)=(224//4, 224//4)=(56, 56)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]   # H//Ph * W//Pw = 56 * 56 = 3136 总的patches数/总的tokens数
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_size, kernel_size=self.patch_size, stride=self.patch_size)
        # 经过self.proj后特征图大小变化为(BS, C, H, W)=(BS, 3, 224, 224)->(BS, embed_size, H//Ph, W//Pw)=(BS, 96, 56, 56)
        if norm_layer is not None:
            self.norm = norm_layer(embed_size)   # nn.LayerNorm(embed_size=96)
        else:
            self.norm = None


    def forward(self, x):
        """x为输入图片 (BS, C, H, W)"""
        BS, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)    # (BS, H//Ph*W//Pw, embed_size)
        # (BS, C, H, W)->(BS, embed_size, H//patch_size, W//patch_size)->(BS, embed_size, H//Ph*W//Pw)->(BS, H//Ph*W//Pw, embed_size)
        if self.norm is not None:
            x = self.norm(x)  # (BS, H//Ph*W//Pw, embed_size)

        return x   # (BS, L=H//Ph*W//Pw, embed_size)



class SwinTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,  # 输入图像的大小
            patch_size=4,  # patch的大小
            in_channels=3,  # 输入图像的通道数
            embed_size=96,  # 输入特征图每个patch/token的嵌入维度
            norm_layer=nn.LayerNorm,  # 层归一化方式
            num_heads=8,  # 多头注意力的头数
            depths=(2, 2, 6, 2),  # 共有4个stage，这4个stage中包含的SwinTransformerBlock的个数分别为2, 2, 6, 2
            # window_size=7,  # 窗口的大小，默认窗口中包含7*7个tokens
            # shift_size=0,  # 移位的大小，一般为窗口大小的一半即window_size//2=3
            # mlp_ratio=4,  # MLP中隐藏层相对于embed_size的倍数
            # qkv_bias=True,  # qkv的线性变换中的偏置
            # qk_scale=None,  # 对Q(K^T)进行缩放的量，其实就是1/sqrt(head_dim)
            # drop=0,  # 模块中使用的dropout
            # attn_drop=0,  # 注意力中使用的dropout
            # drop_path=0,  #
            # act_layer=nn.GELU,  # MLP中使用的激活函数
            # # norm_layer=nn.LayerNorm,  # 模块中使用的层归一化函数
            # fused_window_process=False,  #
            downsample=PatchMerging,
    ):
        super(SwinTransformer, self).__init__()



        self.num_stages = len(depths)  # 4，表示共有4个stage
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_size=embed_size, norm_layer=norm_layer)
        self.input_resolution = self.patch_embed.patches_resolution  # 经过patch_embed后的特征图分辨率  (img_size[0]//patch_size[0], img_size[1]//patch_size[1])=(224//4, 224//4)=(56, 56)
        self.num_patches = self.patch_embed.num_patches  # num_patches 也可以用self.input_resolution[0]*self.input_resolution[1]来代替
        # 需要构造绝对位置编码，加到经过PatchEmbed的输入上 经过PatchEmbed的输入为(BS, num_patches, embed_size)   绝对位置编码是可学习的参数
        self.absolute_pos_embed = nn.Parameter(torch.randn(1, self.input_resolution[0]*self.input_resolution[1], embed_size), requires_grad=True)  # (1, num_patches, embed_size)
        # nn.Parameter将这个张量注册为模型的可学习参数，这意味着这个张量会在训练过程中被不断优化
        # 将不可训练的张量转换成可以训练的参数，并将这个参数绑定到这个模型中


        self.stages = nn.ModuleList()
        # 构造第0,1,2,3个stage，注意每个stage输入的特征图的大小和通道数是不一样的，特征图的高宽逐渐减半，通道数逐渐增大2倍，但最后一个stage即stage3是不改变特征图形状的
        # 输入特征图大小(以高宽中的一个为例):
        # input_resolution[0] -> stage0 -> input_resolution[0]//2 -> stage1 -> input_resolution[0]//4 -> stage2 -> input_resolution[0]//8 -> stage3 -> input_resolution[0]//8
        # 输入特征图的通道数:
        # embed_size -> stage0 -> 2*embed_size -> stage1 -> 4*embed_size -> stage2 -> 8*embed_size -> stage3 -> 8*embed_size
        for i_stage in range(self.num_stages):   # i_stage: 0,1,2,3 分别表示第0,1,2,3个stage
            stage = BasicLayer(
                embed_size=(2**i_stage)*embed_size,
                input_resolution=(self.input_resolution[0]//(2**i_stage), self.input_resolution[1]//(2**i_stage)),
                num_heads=num_heads,
                depth=depths[i_stage],
                downsample=downsample if i_stage < (self.num_stages-1) else None,  # 只有最后一个stage不需要PatchMerging
            )
            self.stages.append(stage)


    def forward(self, x):  # x: (BS, C, H, W)
        x = self.patch_embed(x)  # (BS, C, H, W)->(BS, num_patches=H//Ph*W//Pw, embed_size) (BS, 3, 224, 224)->(BS, 56*56, 96)
        x = x + self.absolute_pos_embed  # (BS, num_patches, embed_size) + (1, num_patches, embed_size) -> (BS, num_patches, embed_size)
        for stage in self.stages:
            x = stage(x)

        # 如果要分类、检测、分割在后面增加对应的头即可
        return x  # (BS, 49, 768)
        # (BS, 56*56, 96)->stage0->(BS, 28*28, 192)->stage1->(BS, 14*14, 384)->stage2->(BS, 7*7, 768)->stage3->(BS, 7*7, 768)



model = SwinTransformer()
x = torch.randn(32, 3, 224, 224)
print(model(x).shape)  # 应该是(BS, 49, 768)





