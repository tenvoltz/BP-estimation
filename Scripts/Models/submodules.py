import torch
import torch.nn as nn
import torchinfo
from dotenv import load_dotenv
import os

from Models.transformer import SimpleMultiHeadedAttention
from einops.layers.torch import Rearrange
from Models.IMSF import ECA

class Downsampling(nn.Module):
    def __init__(self, input_channel, output_channel, input_feature, output_feature):
        super(Downsampling, self).__init__()
        assert input_feature % output_feature == 0
        self.conv = nn.Conv1d(input_channel, output_channel, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(input_feature // output_feature)
        self.bn = nn.BatchNorm1d(output_channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Upsampling(nn.Module):
    def __init__(self, input_channel, output_channel, input_feature, output_feature):
        super(Upsampling, self).__init__()
        assert output_feature % input_feature == 0
        self.conv = nn.Conv1d(input_channel, output_channel, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=output_feature // input_feature)
        self.bn = nn.BatchNorm1d(output_channel)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Excitation_Block(nn.Module):
    def __init__(self, cnn_channel, transformer_channel, reduction=16):
        super(Excitation_Block, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(1)
        channel = cnn_channel + transformer_channel
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(channel, transformer_channel, kernel_size=1),
            nn.BatchNorm1d(transformer_channel),
            nn.Sigmoid(),
        )
    def forward(self, cnn_output, transformer_output):
        x1 = self.avg_pool1(cnn_output)
        x2 = self.avg_pool2(transformer_output)
        x = torch.cat([x1, x2], dim=1)
        x = self.fc(x)
        x = x.unsqueeze(-1)
        x = self.conv(x)
        x = transformer_output * x.expand_as(transformer_output)
        return x


class ChannelShuffle1D(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle1D, self).__init__()
        self.groups = groups
    def forward(self, x):
        batch_size, num_channels, length = x.data.size()
        assert (num_channels % self.groups == 0)
        # Reshape to (b * c // g, g, l)
        x = x.reshape(batch_size * num_channels // self.groups, self.groups, length)
        # Transpose to (g, b * c // g, l)
        x = x.permute(1, 0, 2)
        # Reshape to (g, b, c // g, l)
        x = x.reshape(self.groups, -1, num_channels // self.groups, length)
        return x[0], x[1]

def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]

class GatedConv_Block(nn.Module):
    def __init__(self, input_channel):
        super(GatedConv_Block, self).__init__()
        self.norm = nn.BatchNorm1d(input_channel)
        self.spatial_proj = nn.Conv1d(input_channel,
                                      input_channel,
                                      kernel_size=3,
                                      padding=1,
                                      groups=input_channel)
    def forward(self, identity, gating):
        x = self.norm(gating)
        x = self.spatial_proj(x)
        x = x * identity
        return x


class ConvPatchify_Block(nn.Module):
    def __init__(self, patch_settings, embedding_size):
        super(ConvPatchify_Block, self).__init__()
        # Patch_settings contain the following:
        # kernel_size, input_channel, output_channel
        patchs = []
        for i, patch_setting in enumerate(patch_settings):
            kernel_size, stride, padding, input_channel = patch_setting
            _, _, _, output_channel = patch_settings[i + 1] \
                if i + 1 < len(patch_settings) \
                else (0, 0, 0, embedding_size)
            patchs.append(nn.Conv1d(input_channel, output_channel,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding))
            patchs.append(nn.BatchNorm1d(output_channel))
            patchs.append(nn.ReLU())
        patchs.append(Rearrange('batch channel length -> batch length channel'))
        self.patch = nn.Sequential(*patchs)
    def forward(self, x):
        x = self.patch(x)
        return x

class LCA(nn.Module):
    def __init__(self, channel):
        super(LCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=1),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=1),
            nn.BatchNorm1d(channel),
            nn.Sigmoid(),
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y.expand_as(x)
class DConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(DConv1d, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        return self.conv(x)
class ESPResidual_Block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ESPResidual_Block, self).__init__()
        out_n = int(output_channel / 5)
        out_1 = output_channel - 4 * out_n
        self.resize = nn.Conv1d(input_channel, out_n, 1)
        self.d1 = nn.Sequential(
            nn.Conv1d(out_n, out_1, kernel_size=3, padding='same', dilation=1),
            nn.BatchNorm1d(out_1),
            nn.ReLU(),
        )
        self.d2 = nn.Sequential(
            nn.Conv1d(out_n, out_n, kernel_size=3, padding='same', dilation=2),
            #nn.LayerNorm([out_n] + [input_length]),
            nn.BatchNorm1d(out_n),
            nn.ReLU(),
        )
        self.d4 = nn.Sequential(
            nn.Conv1d(out_n, out_n, kernel_size=3, padding='same', dilation=4),
            #nn.LayerNorm([out_n] + [input_length]),
            nn.BatchNorm1d(out_n),
            nn.ReLU(),
        )
        self.d8 = nn.Sequential(
            nn.Conv1d(out_n, out_n, kernel_size=3, padding='same', dilation=8),
            #nn.LayerNorm([out_n] + [input_length]),
            nn.BatchNorm1d(out_n),
            nn.ReLU(),
        )
        self.d16 = nn.Sequential(
            nn.Conv1d(out_n, out_n, kernel_size=3, padding='same', dilation=16),
            #nn.LayerNorm([out_n] + [input_length]),
            nn.BatchNorm1d(out_n),
            nn.ReLU(),
        )
        self.fusion = nn.Conv1d(out_n * 4, out_n * 4, 3, padding='same')
        self.bn = nn.Sequential(
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
        )
        self.residual = nn.Conv1d(input_channel, output_channel, 1)
    def forward(self, x):
        # Save the input for the residual connection
        residual = self.residual(x)
        x = self.resize(x)
        d1 = self.d1(x)
        d2 = self.d2(x)
        d4 = self.d4(x)
        d8 = self.d8(x)
        d16 = self.d16(x)

        fusion = torch.cat([d2, d4, d8, d16], dim=1)
        fusion = self.fusion(fusion)

        #add1 = d2
        #add2 = add1 + d4
        #add3 = add2 + d8
        #add4 = add3 + d16

        x = torch.cat([d1, fusion], dim=1)
        x = residual + x
        x = self.bn(x)
        return x

class RandomEncoding(nn.Module):
    def __init__(self, sequence_size, embedding_size):
        super(RandomEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.encoding = nn.Parameter(torch.randn(1, sequence_size, embedding_size))
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]
class LiPatch_Block(nn.Module):
    def __init__(self, embedding_size, patch_size, input_channel):
        super(LiPatch_Block, self).__init__()
        patch_dim = input_channel * patch_size

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embedding_size),
            nn.LayerNorm(embedding_size),
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        return x

class Patchify_Block(nn.Module):
    def __init__(self, patch_size, input_channel, output_channel):
        super(Patchify_Block, self).__init__()
        self.patch_size = patch_size
        # Depthwise convolution
        self.patch = nn.LazyConv1d(input_channel,
                                   kernel_size=patch_size,
                                   stride=patch_size,
                                   groups=input_channel)
        self.expand = nn.Conv1d(input_channel, output_channel, 1)
        #self.bn = nn.BatchNorm1d(output_channel)
    def forward(self, x):
        x = self.patch(x)
        x = self.expand(x)
        #x = self.bn(x)
        return x
class MSCA_Block(nn.Module):
    def __init__(self, b1, b2, b3, b4):
        super(MSCA_Block, self).__init__()
        self.conv3x3 = nn.Sequential(
            nn.LazyConv1d(b1, kernel_size=3, padding=1),
            nn.BatchNorm1d(b1),
            nn.ReLU(),
        )
        self.conv11x11 = nn.Sequential(
            nn.LazyConv1d(b2, kernel_size=11, padding=5),
            nn.BatchNorm1d(b2),
            nn.ReLU(),
        )
        self.conv7x7 = nn.Sequential(
            nn.LazyConv1d(b3, kernel_size=7, padding=3),
            nn.BatchNorm1d(b3),
            nn.ReLU(),
        )
        self.conv9x9 = nn.Sequential(
            nn.LazyConv1d(b4, kernel_size=9, padding=4),
            nn.BatchNorm1d(b4),
            nn.ReLU(),
        )
    def forward(self, x):
        x3 = self.conv3x3(x)
        x11 = self.conv11x11(x)
        x7 = self.conv7x7(x)
        x9 = self.conv9x9(x)
        x = torch.cat([x3, x7, x9, x11], dim=1)
        return x
class MHA_MixerBlock(nn.Module):
    def __init__(self, signal_amount=3, sequence_length=1024, embedding_size=64, dropout=0.1):
        super(MHA_MixerBlock, self).__init__()
        self.signal_amount = signal_amount
        # MHA amount is the amount of possible combinations of signals
        mha_amount = signal_amount
        self.MHAs = nn.ModuleList([SimpleMultiHeadedAttention(sequence_length=sequence_length,
                                                             head_amount=4,
                                                             embedding_size=embedding_size,
                                                             dropout=dropout)
                                  for _ in range(mha_amount)])
        self.MaxPools = nn.ModuleList([nn.MaxPool1d(kernel_size=signal_amount) for _ in range(signal_amount)])

    def forward(self, x):
        # x shape: (batch_size, signal_amount, sequence_length, embedding_size)
        for i in range(1):
            mha_output = []
            for j in range(self.signal_amount):
                j = (i + j) % self.signal_amount
                mha_output.append(self.MHAs[i * self.signal_amount + j](x[:, j], x[:, i], x[:, i]))
            mha_output = torch.concat(mha_output, dim=-1)
            x[:, i] = self.MaxPools[i](mha_output)
        return x

class ModifiedIMSFBlock(nn.Module):
    def __init__(self, low_level_channel, high_level_channel, fusion_channel):
        super(ModifiedIMSFBlock, self).__init__()
        self.conv5x5 = nn.Sequential(
            nn.LazyConv1d(high_level_channel, kernel_size=3, padding=1),
            nn.LazyConv1d(high_level_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(high_level_channel),
            nn.ReLU(),
        )
        self.conv9x9 = nn.Sequential(
            nn.LazyConv1d(high_level_channel, kernel_size=3, dilation=2, padding=2),
            nn.LazyConv1d(high_level_channel, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(high_level_channel),
            nn.ReLU(),
        )
        self.conv11x11 = nn.Sequential(
            nn.LazyConv1d(high_level_channel, kernel_size=3, padding=1),
            nn.LazyConv1d(high_level_channel, kernel_size=3, dilation=2, padding=2),
            nn.LazyConv1d(high_level_channel, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(high_level_channel),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.LazyConv1d(fusion_channel, kernel_size=3, padding=4),
            nn.BatchNorm1d(fusion_channel),
            nn.ReLU(),
        )
        self.conv3x3 = nn.Sequential(
            nn.LazyConv1d(low_level_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(low_level_channel),
            nn.ReLU(),
        )
    def forward(self, x):
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x9 = self.conv9x9(x)
        x11 = self.conv11x11(x)
        x = torch.cat([x5, x9, x11], dim=1)
        x = self.fusion(x)
        x = torch.cat([x, x3], dim=1)
        return x

class DiCNN_Block(nn.Module):
    def __init__(self, b1, b2, b3, b4):
        super(DiCNN_Block, self).__init__()
        self.conv3x3 = nn.Sequential(
            nn.LazyConv1d(b1, kernel_size=3, padding=1),
            nn.BatchNorm1d(b1),
            nn.ReLU(),
        )
        self.conv5x5 = nn.Sequential(
            nn.LazyConv1d(b2, kernel_size=3, padding=1),
            nn.LazyConv1d(b2, kernel_size=3, padding=1),
            nn.BatchNorm1d(b2),
            nn.ReLU(),
        )
        self.conv9x9 = nn.Sequential(
            nn.LazyConv1d(b3, kernel_size=3, dilation=2, padding=2),
            nn.LazyConv1d(b3, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(b3),
            nn.ReLU(),
        )
        self.conv11x11 = nn.Sequential(
            nn.LazyConv1d(b4, kernel_size=3, padding=1),
            nn.LazyConv1d(b4, kernel_size=3, dilation=2, padding=2),
            nn.LazyConv1d(b4, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(b4),
            nn.ReLU(),
        )
    def forward(self, x):
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x9 = self.conv9x9(x)
        x11 = self.conv11x11(x)
        x = torch.cat([x3, x5, x9, x11], dim=1)
        return x


if __name__ == '__main__':
    dummy_input = {'signals': torch.randint(0, 10, (2, 2, 3))}
    print(dummy_input)
    model = ChannelShuffle1D(2)
    x1, x2 = model(dummy_input['signals'])
    print(x1)
    print(x2)
