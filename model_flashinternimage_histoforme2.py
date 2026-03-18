import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from einops import rearrange

from myFFCResblock0 import myFFCResblock
from FlashInternImage.models.flash_intern_image import FlashInternImage


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class DWT_transform(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv1x1_low = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency, dwt_high_frequency


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))

    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))

    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))

    return block


##########################################################################
## Histoformer bottleneck block
##########################################################################

Conv2d = nn.Conv2d


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        import numbers
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        import numbers
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        raw_hidden = dim * ffn_expansion_factor

        # Make hidden_features divisible by 4 so:
        # 1) project_in output works with PixelShuffle(2)
        # 2) post-shuffle channels split evenly
        # 3) dwconv branch dimensions stay consistent
        hidden_features = max(4, int(round(raw_hidden / 4.0) * 4))

        self.hidden_features = hidden_features

        self.project_in = Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        branch_channels = hidden_features // 4

        self.dwconv_5 = Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=branch_channels,
            bias=bias
        )
        self.dwconv_dilated2_1 = Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            groups=branch_channels,
            bias=bias,
            dilation=2
        )

        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)

        x1, x2 = x.chunk(2, dim=1)

        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)

        x = F.mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)

        return x


class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(
            dim * 5,
            dim * 5,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 5,
            bias=bias
        )
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)

        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"

        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)

        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        x_work = x.clone()

        x_sort, idx_h = x_work[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x_work[:, :c // 2] = x_sort

        qkv = self.qkv_dwconv(self.qkv(x_work))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)

        out = out1 * out2
        out = self.project_out(out)

        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace

        return out


class HistoTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(HistoTransformerBlock, self).__init__()

        self.attn_g = Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        x = x + self.ffn(self.norm_ff1(x))
        return x


##########################################################################
## DWT + FFC U-Net branch with Histoformer block at bottleneck
##########################################################################

class dwt_ffc_UNet2_Histoformer(nn.Module):
    def __init__(self, output_nc=3, nf=16):
        super(dwt_ffc_UNet2_Histoformer, self).__init__()

        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(16, nf - 1, 4, 2, 1, bias=False))

        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf * 2 - 2, name, transposed=False, bn=True, relu=False, dropout=False)

        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf * 2, nf * 4 - 4, name, transposed=False, bn=True, relu=False, dropout=False)

        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf * 4, nf * 8 - 8, name, transposed=False, bn=True, relu=False, dropout=False)

        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf * 8, nf * 8 - 16, name, transposed=False, bn=True, relu=False, dropout=False)

        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf * 4, nf * 4, name, transposed=False, bn=False, relu=False, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer6 = blockUNet(nf * 4, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer5 = blockUNet(nf * 16 + 16, nf * 8, name, transposed=True, bn=True, relu=True, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer4 = blockUNet(nf * 16 + 8, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer3 = blockUNet(nf * 8 + 4, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer2 = blockUNet(nf * 4 + 2, nf, name, transposed=True, bn=True, relu=True, dropout=False)

        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2 + 1, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        self.initial_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = layer1
        self.DWT_down_0 = DWT_transform(3, 1)

        self.layer2 = layer2
        self.DWT_down_1 = DWT_transform(16, 2)

        self.layer3 = layer3
        self.DWT_down_2 = DWT_transform(32, 4)

        self.layer4 = layer4
        self.DWT_down_3 = DWT_transform(64, 8)

        self.layer5 = layer5
        self.DWT_down_4 = DWT_transform(128, 16)

        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

        self.tail_conv1 = nn.Conv2d(48, 32, 3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32)
        self.tail_conv2 = nn.Conv2d(nf * 2, output_nc, 3, padding=1, bias=True)

        self.FFCResNet = myFFCResblock(input_nc=64, output_nc=64)

        self.HistoBlock = HistoTransformerBlock(
            dim=64,
            num_heads=8,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias'
        )

    def forward(self, x):
        conv_start = self.initial_conv(x)
        conv_start = self.bn1(conv_start)

        conv_out1 = self.layer1(conv_start)
        dwt_low_0, dwt_high_0 = self.DWT_down_0(x)
        out1 = torch.cat([conv_out1, dwt_low_0], 1)

        conv_out2 = self.layer2(out1)
        dwt_low_1, dwt_high_1 = self.DWT_down_1(out1)
        out2 = torch.cat([conv_out2, dwt_low_1], 1)

        conv_out3 = self.layer3(out2)
        dwt_low_2, dwt_high_2 = self.DWT_down_2(out2)
        out3 = torch.cat([conv_out3, dwt_low_2], 1)

        out3_ffc = self.FFCResNet(out3)
        out3_histo = self.HistoBlock(out3_ffc)

        dout3 = self.dlayer6(out3_histo)

        Tout3_out2 = torch.cat([dout3, out2, dwt_high_1], 1)
        Tout2 = self.dlayer2(Tout3_out2)

        Tout2_out1 = torch.cat([Tout2, out1, dwt_high_0], 1)
        Tout1 = self.dlayer1(Tout2_out1)

        Tout1_outinit = torch.cat([Tout1, conv_start], 1)
        tail1 = self.tail_conv1(Tout1_outinit)
        tail2 = self.bn2(tail1)
        dout1 = self.tail_conv2(tail2)

        return dout1


##########################################################################
## Existing attention blocks for FlashInternImage branch
##########################################################################

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class knowledge_adaptation_convnext(nn.Module):
    def __init__(self):
        super(knowledge_adaptation_convnext, self).__init__()
        self.encoder = FlashInternImage()

        checkpoint_model = torch.load(
            'flash_intern_image_l_22kto1k_384.pth',
            map_location='cpu'
        )['model']

        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if 'conv_head' in k:
                del checkpoint_model[k]
                print(k)
            elif k in ['head.weight', 'head.bias']:
                del checkpoint_model[k]
                print(k)
            else:
                continue

        try:
            self.encoder.load_state_dict(checkpoint_model, strict=True)
            print('Loading FlashInterimage success')
        except Exception as e:
            print('Loading FlashInterimage error')
            print(e)

        self.up_block = nn.PixelShuffle(2)
        self.up_block2 = nn.PixelShuffle(4)

        self.attention0 = CP_Attention_block(default_conv, 1280, 3)
        self.attention1 = CP_Attention_block(default_conv, 320, 3)
        self.attention2 = CP_Attention_block(default_conv, 240, 3)
        self.attention3 = CP_Attention_block(default_conv, 140, 3)
        self.attention4 = CP_Attention_block(default_conv, 32, 3)
        self.attention5 = CP_Attention_block(default_conv, 8, 3)

        self.conv_ = nn.Conv2d(140, 128, kernel_size=3, padding=1)
        self.conv_process_1 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.conv_process_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        
    def forward(self, input):
        x_layer1, x_layer2, x_layer3 = self.encoder(input)
        x_mid = self.attention0(x_layer3)

        x = self.up_block(x_mid)
        x = self.attention1(x)

        x = torch.cat((x, x_layer2), 1)

        x = self.up_block(x)
        x = self.attention2(x)

        x = torch.cat((x, x_layer1), 1)

        x = self.up_block(x)
        x = self.attention3(x)

        x = self.conv_(x)

        x = self.up_block(x)
        x = self.attention4(x)

        x = self.up_block(x)
        x = self.attention5(x)

        x = self.conv_process_1(x)
        out = self.conv_process_2(x)
        return out


##########################################################################
## Fusion model
##########################################################################

class fusion_net_histoformer2(nn.Module):
    def __init__(self):
        super(fusion_net_histoformer2, self).__init__()
        self.dwt_branch = dwt_ffc_UNet2_Histoformer()
        self.knowledge_adaptation_branch = knowledge_adaptation_convnext()
        self.fusion = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(11, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, input):
        dwt_branch = self.dwt_branch(input)
        knowledge_adaptation_branch = self.knowledge_adaptation_branch(input)
        x = torch.cat([dwt_branch, knowledge_adaptation_branch], 1)
        x = self.fusion(x)
        return x


##########################################################################
## Discriminator
##########################################################################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))