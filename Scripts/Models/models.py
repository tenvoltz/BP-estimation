import torch
import torch.nn as nn
import torchinfo
from dotenv import load_dotenv
import os

from einops.layers.torch import Rearrange
from Models.FNet import *
from Models.submodules import *
from Models.IMSF import ECA, IMSF_Block
from Models.transformer import *
from Models.mobilenetv2 import MobileNetV2

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

SIGNALS_LIST = [signal.strip().lower() for signal in os.getenv('SIGNALS').split(',')]
SIGNAL_LENGTH = int(os.getenv('INPUT_LENGTH'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))


class TransFMSD(nn.Module):
    def __init__(self, bias_init=None):
        super(TransFMSD, self).__init__()
        cnn_blocks_setting = [
            # channel, reduction
            [32, 2],
            [40, 2],
            [64, 2],
            [80, 2],
            [160, 2]
        ]
        patch_blocks_setting = [
            # kernel_size, stride, padding, in_channel, out_channel
            [3, 2, 1, 48],  # 512 out   # 625
            [3, 2, 1, 32],  # 256 out   # 312
            [3, 2, 1, 64],  # 128_out   # 156
            # [3, 2, 1, 96],
        ]
        transformer_blocks_setting = [
            # patch_amount, reduction, head_amount, mlp_hidden
            [128, 1, 8, 256],
            [128, 2, 8, 256],
            [64, 1, 8, 256],
            [64, 1, 8, 256],
        ]
        cnn_branch = []
        transformer_branch = []
        fusion_branch = []
        flatten_branch = []

        embedding_size = SIGNAL_LENGTH
        cnn_channel = len(SIGNALS_LIST)
        for i in range(len(cnn_blocks_setting)):
            out_channel, reduction = cnn_blocks_setting[i]
            embedding_size //= reduction
            cnn = [
                ESPResidual_Block(cnn_channel, out_channel),
                LCA(out_channel),
                nn.Conv1d(out_channel, out_channel, kernel_size=1),
                nn.BatchNorm1d(out_channel),
            ]
            if reduction != 1:
                cnn.insert(1, nn.MaxPool1d(reduction))
            cnn = nn.Sequential(*cnn)
            cnn_branch.append(cnn)
            cnn_channel = out_channel

        flatten_branch.append(nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1)
        ))

        transformer_channel = len(SIGNALS_LIST)
        embedding_size = 128
        for i in range(len(transformer_blocks_setting)):
            patch_amount, reduction, head_amount, mlp_hidden = transformer_blocks_setting[i]

            # Construct a diagonal mask
            mask = 1 - torch.eye(patch_amount)
            mask = mask.type(torch.bool)

            transformer = [
                MetaFormerEncoderBlock(
                    # channel_mixer=MultiHeadedPoolingAttention(head_amount,
                    #                                          embedding_size,
                    #                                          reduction_ratio=reduction,
                    #                                          dropout=0.1),
                    channel_mixer=MultiHeadedAttention(head_amount, embedding_size,
                                                       dropout=0.1, attention=LearnableScalingAttention,
                                                       mask=mask)
                    if head_amount != None else FNetBlock(),
                    feature_mixer=LocalityFeedForward(embedding_size, head_amount),
                    #feature_mixer=PositionwiseFeedForward(embedding_size, mlp_hidden),
                    sequence_length=patch_amount,
                    embedding_size=embedding_size,
                    dropout=0.1
                ),
            ]
            if i == 0:
                # transformer.insert(0, LiPatch_Block(embedding_size=embedding_size,
                #                                    patch_size=SIGNAL_LENGTH // patch_amount,
                #                                    input_channel=transformer_channel))
                shift_amount = 16
                transformer.insert(0, LocalityInvariantShifting([i * (SIGNAL_LENGTH // shift_amount)
                                                                 for i in range(shift_amount)]))
                transformer.insert(1, ConvPatchify_Block(patch_blocks_setting, embedding_size))
                # transformer.insert(2, PositionalEncoding(embedding_size))
                transformer.insert(2, RandomEncoding(patch_amount, embedding_size))
            else:
                transformer.insert(0, Rearrange('batch embedding patch -> batch patch embedding'))
            if reduction != 1:
                transformer.append(SequencePooling1D(reduction))
            transformer.append(Rearrange('batch patch embedding -> batch embedding patch'))
            transformer = nn.Sequential(*transformer)
            transformer_branch.append(transformer)

            transformer_channel = embedding_size

        flatten_branch.append(nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1)
        ))

        self.cnn_branch = nn.ModuleList(cnn_branch)
        self.transformer_branch = nn.ModuleList(transformer_branch)
        self.fusion_branch = nn.ModuleList(fusion_branch)
        self.flatten = nn.ModuleList(flatten_branch)

        regression_channel = transformer_channel + cnn_channel
        regression_head = nn.Linear(regression_channel, 2)
        if bias_init is not None:
            with torch.no_grad():
                regression_head.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            regression_head,
            nn.ReLU()
        )

    def forward(self, x):
        transformer_output = x['signals']
        cnn_output = x['signals']
        mixer_output = []
        for i in range(len(self.cnn_branch)):
            cnn_output = self.cnn_branch[i](cnn_output)
        for i in range(len(self.transformer_branch)):
            transformer_output = self.transformer_branch[i](transformer_output)
        cnn_output = self.flatten[0](cnn_output)
        transformer_output = self.flatten[1](transformer_output)
        # Concatenate all output
        x = torch.cat((cnn_output, transformer_output), dim=1)
        x = self.regression_head(x)
        return x

class Transformer_Only(nn.Module):
    # Change this to Transformer_Only later on
    def __init__(self, bias_init=None):
        super(Transformer_Only, self).__init__()
        patch_blocks_setting = [
            # kernel_size, stride, padding, in_channel, out_channel
            [3, 2, 1, 48],                  # 512 out
            [3, 2, 1, 64],                  # 256 out
            [3, 2, 1, 96],                  # 128_out
            #[3, 2, 1, 96],
        ]
        transformer_blocks_setting = [
            # patch_amount, reduction, head_amount, mlp_hidden
            [128, 1, 8, 256],
            [128, 2, 8, 256],
            [64, 1, 8, 256],
            [64, 1, 8, 256],
        ]
        regression_channel = 0
        transformer_branch = []
        in_channel = len(SIGNALS_LIST)
        embedding_size = 128
        for i in range(len(transformer_blocks_setting)):
            patch_amount, reduction, head_amount, mlp_hidden = transformer_blocks_setting[i]

            # Construct a diagonal mask
            mask = 1 - torch.eye(patch_amount)
            mask = mask.type(torch.bool)

            transformer = [
                MetaFormerEncoderBlock(
                    #channel_mixer=MultiHeadedPoolingAttention(head_amount,
                    #                                          embedding_size,
                    #                                          reduction_ratio=reduction,
                    #                                          dropout=0.1),
                    channel_mixer=MultiHeadedAttention(head_amount, embedding_size,
                                                       dropout=0.1, attention=LearnableScalingAttention,
                                                       mask=mask)
                    if head_amount != None else FNetBlock(),
                    feature_mixer=LocalityFeedForward(embedding_size, head_amount),
                    #feature_mixer=PositionwiseFeedForward(embedding_size, mlp_hidden),
                    sequence_length=patch_amount,
                    embedding_size=embedding_size,
                    dropout=0.1
                ),
            ]
            if i == 0:
                # transformer.insert(0, LiPatch_Block(embedding_size=embedding_size,
                #                                    patch_size=SIGNAL_LENGTH // patch_amount,
                #                                    input_channel=transformer_channel))
                shift_amount = 16
                transformer.insert(0, LocalityInvariantShifting([i * (SIGNAL_LENGTH // shift_amount)
                                                                 for i in range(shift_amount)]))
                transformer.insert(1, ConvPatchify_Block(patch_blocks_setting, embedding_size))
                transformer.insert(2, PositionalEncoding(embedding_size))
            if reduction != 1:
                transformer.append(SequencePooling1D(reduction))

            transformer = nn.Sequential(*transformer)
            transformer_branch.append(transformer)

            in_channel = patch_amount // reduction

        regression_channel += in_channel

        #transformer_branch.append(nn.Sequential(
        #    nn.AvgPool1d(kernel_size=embedding_size),
        #    nn.Flatten(),
        #    nn.Dropout(0.1)
        #))
        self.flatten = nn.Sequential(
            nn.AvgPool1d(kernel_size=embedding_size),
            nn.Flatten(),
            nn.Dropout(0.1)
        )

        self.transformer_branch = nn.ModuleList(transformer_branch)

        regression_head = nn.Linear(regression_channel, 2)
        if bias_init is not None:
            with torch.no_grad():
                regression_head.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            regression_head,
            nn.ReLU()
        )

    def forward(self, x):
        x = x['signals']
        #transformer_output = []
        for i in range(len(self.transformer_branch)):
            x = self.transformer_branch[i](x)
            #transformer_output.append(self.transformer_branch[i](x))
        # Concatenate all output
        #x = torch.cat(transformer_output, dim=1)
        x = self.flatten(x)
        x = self.regression_head(x)
        return x


"""
class IMSF_Net(nn.Module):
    # Change this to MSCA_FNet later on, Benchmark model
    def __init__(self, bias_init=None):
        super(IMSF_Net, self).__init__()
        cnn_blocks_setting = [
            # channel, reduction, head_amount
            [32, 2, 4],
            [40, 2, 4],
            [64, 2, 4],
            [80, 2, 4],
            [160, 2, 4],
            # b1, b2, b3, b4
            #[6, 24, 18, 0],
            #[64, 64, 192, 0],
            # h, l, fu
            #[6, 24, 18],
            #[64, 64, 192],
        ]
        cnn_branch = []
        transformer_branch = []
        fusion_branch = []

        in_channel = len(SIGNALS_LIST)
        embedding_size = SIGNAL_LENGTH
        for i in range(len(cnn_blocks_setting)):
            #b1, b2, b3, b4 = cnn_blocks_setting[i]
            #out_channel = b1 + b2 + b3 + b4
            #h, l, fu = cnn_blocks_setting[i]
            #out_channel = l + fu
            out_channel, reduction, head_amount = cnn_blocks_setting[i]
            embedding_size //= reduction
            cnn = nn.Sequential(
                ESPResidual_Block(in_channel, out_channel),
                nn.MaxPool1d(reduction),
                LCA(out_channel),
            )
            '''
            cnn = nn.Sequential(
                MSCA_Block(b1, b2, b3, b4), # Now with 11x11 to replace 5x5
                nn.MaxPool1d(2),
                ECA(out_channel)
            )
            '''
            '''
            cnn = nn.Sequential(
                ModifiedIMSFBlock(l, h, fu),
                nn.MaxPool1d(2),
                ECA(out_channel)
            )
            '''
            # CNN_branch output shape: (batch_size, b1 + b2 + b3 + b4, embedding_size)
            cnn_branch.append(cnn)


            transformer = [
                Patchify_Block(reduction, in_channel, out_channel),
                #Rearrange('b c e -> b e c'),
                TransformerEncoderBlock(head_amount=head_amount, embedding_size=embedding_size, dropout=0.1),
                #Rearrange('b e c -> b c e'),
            ]
            if i == 0:
                transformer.insert(1, PositionalEncoding(embedding_size))
            transformer = nn.Sequential(*transformer)
            transformer_branch.append(transformer)

            fusion = nn.Sequential(
                #nn.LazyConv1d(2 * out_channel, kernel_size=9, padding=4, groups=2 * out_channel),
                #LCA(out_channel),
                nn.LazyConv1d(out_channel, kernel_size=1),
                #nn.LayerNorm(out_channel),
                nn.BatchNorm1d(out_channel),
            )
            fusion_branch.append(fusion)
            in_channel = out_channel

        self.cnn_branch = nn.ModuleList(cnn_branch)
        self.transformer_branch = nn.ModuleList(transformer_branch)
        self.fusion_branch = nn.ModuleList(fusion_branch)
        self.flatten = nn.Sequential(
            nn.AvgPool1d(kernel_size=embedding_size),
            nn.Flatten(),
        )

        regression_head = nn.Linear(in_channel, 2)
        if bias_init is not None:
            with torch.no_grad():
                regression_head.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            regression_head,
            nn.ReLU()
        )

    def forward(self, x):
        x = x['signals']
        cnn_output = x
        for i in range(len(self.cnn_branch)):
            cnn_output = self.cnn_branch[i](cnn_output)
            transformer_output = self.transformer_branch[i](x)
            # Concatenate two layer then put through fusion
            x = torch.cat((cnn_output, transformer_output), dim=1)
            x = self.fusion_branch[i](x)
        x = self.flatten(x)
        x = self.regression_head(x)
        return x
"""
#'''
class IMSF_Net(nn.Module):
    # Change this to MSCA_FNet later on
    def __init__(self, bias_init=None):
        super(IMSF_Net, self).__init__()
        cnn_blocks_setting = [
            # channel, reduction
            [32, 2],
            [40, 2],
            [64, 2],
            [80, 2],
            [160, 2]
        ]
        patch_blocks_setting = [
            # kernel_size, stride, padding, in_channel, out_channel
            [3, 2, 1, 48],   # 512 out   # 625
            [3, 2, 1, 32],                  # 256 out   # 312
            [3, 2, 1, 64],                  # 128_out   # 156
            #[3, 2, 1, 96],
        ]
        transformer_blocks_setting = [
            # patch_amount, reduction, head_amount, mlp_hidden
            [128, 1, 4, 512],
            [128, 2, 4, 512],
            [64, 1, 4, 512],
            [64, 1, 4, 512],
            [64, 1, 4, 512],
        ]
        cnn_branch = []
        transformer_branch = []
        fusion_branch = []
        flatten_branch = []

        embedding_size = SIGNAL_LENGTH
        cnn_channel = len(SIGNALS_LIST)
        for i in range(len(cnn_blocks_setting)):
            out_channel, reduction = cnn_blocks_setting[i]
            embedding_size //= reduction
            cnn = [
                ESPResidual_Block(cnn_channel, out_channel),
                LCA(out_channel),
            ]
            if reduction != 1:
                cnn.insert(1, nn.MaxPool1d(reduction))
            cnn = nn.Sequential(*cnn)
            cnn_branch.append(cnn)
            cnn_channel = out_channel

        flatten_branch.append(nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1)
        ))

        transformer_channel = len(SIGNALS_LIST)
        embedding_size = 128
        for i in range(len(transformer_blocks_setting)):
            patch_amount, reduction, head_amount, mlp_hidden = transformer_blocks_setting[i]

            # Construct a diagonal mask
            mask = 1 - torch.eye(patch_amount)
            mask = mask.type(torch.bool)

            transformer = [
                MetaFormerEncoderBlock(
                    # channel_mixer=MultiHeadedPoolingAttention(head_amount,
                    #                                          embedding_size,
                    #                                          reduction_ratio=reduction,
                    #                                          dropout=0.1),
                    channel_mixer=MultiHeadedAttention(head_amount, embedding_size,
                                                       dropout=0.1, attention=LearnableScalingAttention,
                                                       mask=mask)
                    if head_amount != None else FNetBlock(),
                    #feature_mixer=LocalityFeedForward(embedding_size, head_amount),
                    feature_mixer=PositionwiseFeedForward(embedding_size, mlp_hidden),
                    sequence_length=patch_amount,
                    embedding_size=embedding_size,
                    dropout=0.1
                ),
            ]
            if i == 0:
                # transformer.insert(0, LiPatch_Block(embedding_size=embedding_size,
                #                                    patch_size=SIGNAL_LENGTH // patch_amount,
                #                                    input_channel=transformer_channel))
                shift_amount = 16
                transformer.insert(0, LocalityInvariantShifting([i * (SIGNAL_LENGTH // shift_amount)
                                                                 for i in range(shift_amount)]))
                transformer.insert(1, ConvPatchify_Block(patch_blocks_setting, embedding_size))
                #transformer.insert(2, PositionalEncoding(embedding_size))
                transformer.insert(2, RandomEncoding(patch_amount, embedding_size))
            else:
                transformer.insert(0, Rearrange('batch embedding patch -> batch patch embedding'))
            if reduction != 1:
                transformer.append(SequencePooling1D(reduction))
            transformer.append(Rearrange('batch patch embedding -> batch embedding patch'))
            transformer = nn.Sequential(*transformer)
            transformer_branch.append(transformer)

            transformer_channel = embedding_size

        flatten_branch.append(nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1)
        ))
        regression_channel = 128

        cnn_feature = SIGNAL_LENGTH
        for i in range(len(transformer_blocks_setting)):
            out_channel, cnn_reduction = cnn_blocks_setting[i]
            patch_amount, patch_reduction, head_amount, mlp_hidden = transformer_blocks_setting[i]
            transformer_feature = patch_amount // patch_reduction
            cnn_feature //= cnn_reduction
            #"""
            if i != len(transformer_blocks_setting) - 1:
                fusion = nn.ModuleList([
                    #ChannelShuffle1D(2),
                    #GatedConv_Block(out_channel),
                    #nn.BatchNorm1d(out_channel),
                    #nn.Sequential(
                    #   nn.Conv1d(out_channel * 2, out_channel, kernel_size=1),
                    #    nn.BatchNorm1d(out_channel),
                    #)
                    #Excitation_Block(out_channel, embedding_size),
                    #Excitation_Block(embedding_size, out_channel),
                    #nn.Sequential(
                    #    nn.LazyLinear(128),
                    #    nn.Dropout(0.1),
                    #    nn.ReLU()
                    #),
                    Downsampling(out_channel, embedding_size, cnn_feature, transformer_feature)
                    if cnn_feature >= transformer_feature
                    else Upsampling(out_channel, embedding_size, cnn_feature, transformer_feature),
                    Downsampling(embedding_size, out_channel, transformer_feature, cnn_feature)
                    if transformer_feature >= cnn_feature
                    else Upsampling(embedding_size, out_channel, transformer_feature, cnn_feature),
                    nn.Sequential(
                        nn.LazyConv1d(regression_channel, kernel_size=1),
                        nn.BatchNorm1d(regression_channel),
                        nn.AvgPool1d(transformer_feature),
                        nn.Flatten(),
                    ),
                    nn.Sequential(
                        nn.LazyConv1d(regression_channel, kernel_size=1),
                        nn.BatchNorm1d(regression_channel),
                        nn.AvgPool1d(cnn_feature),
                        nn.Flatten(),
                    ),

                ])
            #"""
            else:
                fusion = nn.ModuleList([
                    nn.MaxPool1d(cnn_feature // transformer_feature)
                    if cnn_feature >= transformer_feature
                    else nn.Identity(),
                    nn.MaxPool1d(transformer_feature // cnn_feature)
                    if transformer_feature >= cnn_feature
                    else nn.Identity(),
                    nn.Sequential(
                        nn.LazyConv1d(regression_channel, kernel_size=1),
                        nn.BatchNorm1d(regression_channel),
                        nn.AvgPool1d(min(cnn_feature, transformer_feature)),
                        nn.Flatten(),
                    )
                ])
            fusion_branch.append(fusion)

        self.cnn_branch = nn.ModuleList(cnn_branch)
        self.transformer_branch = nn.ModuleList(transformer_branch)
        self.fusion_branch = nn.ModuleList(fusion_branch)
        self.flatten = nn.ModuleList(flatten_branch)

        regression_head = nn.Linear(512, 2)
        if bias_init is not None:
            with torch.no_grad():
                regression_head.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            nn.LazyLinear(512),
            nn.Dropout(0.1),
            nn.ReLU(),
            regression_head,
            nn.ReLU()
        )


    def forward(self, x):
        transformer_output = x['signals']
        cnn_output = x['signals']
        mixer_output = []
        for i in range(len(self.cnn_branch)):
            cnn_output = self.cnn_branch[i](cnn_output)
            transformer_output = self.transformer_branch[i](transformer_output)
            if i < len(self.fusion_branch) - 1:
                #temp, t_excitation = self.fusion_branch[i][0](cnn_output, transformer_output)
                #cnn_output, c_excitation = self.fusion_branch[i][1](transformer_output, cnn_output)
                temp = transformer_output + self.fusion_branch[i][0](cnn_output)
                cnn_output = cnn_output + self.fusion_branch[i][1](transformer_output)
                transformer_output = temp
                #t_excitation = self.fusion_branch[i][2](t_excitation)
                #c_excitation = self.fusion_branch[i][2](c_excitation)
                t_excitation = self.fusion_branch[i][2](transformer_output)
                c_excitation = self.fusion_branch[i][3](cnn_output)
                mixer_output.append(t_excitation + c_excitation)
            elif i == len(self.fusion_branch) - 1:
                cnn_resize, transformer_resize, mixer = self.fusion_branch[i]
                cnn_resize = cnn_resize(cnn_output)
                transformer_resize = transformer_resize(transformer_output)
                mixer_output.append(mixer(torch.cat((cnn_resize, transformer_resize), dim=1)))
        cnn_output = self.flatten[0](cnn_output)
        transformer_output = self.flatten[1](transformer_output)
        # Concatenate all output
        #mixer_output = torch.cat(mixer_output, dim=1)
        mixer_output = torch.stack(mixer_output, dim=1).sum(dim=1)
        x = torch.cat((cnn_output, mixer_output, transformer_output), dim=1)
        x = self.regression_head(x)
        return x
#'''

class Transformer_Basic(nn.Module):
    def __init__(self, bias_init=None):
        super(Transformer_Basic, self).__init__()
        self.embedding = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.transformer = nn.Sequential(
            TransformerEncoder(
                head_amount=4,
                embedding_size=64,
                layer_amount=2,
                dropout=0.1
            ),
            nn.AvgPool1d(kernel_size=64),
            nn.Flatten(),
        )
        BP_regressor = nn.Linear(SIGNAL_LENGTH // 2, 2)
        if bias_init is not None:
            with torch.no_grad():
                BP_regressor.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            BP_regressor,
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embedding(x['signals'])
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.regression_head(x)
        return x


class Transformer_Mixer(nn.Module):
    def __init__(self, signal_amount=3, layer_amount=2):
        super(Transformer_Mixer, self).__init__()
        embedding_block = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.embedding = nn.ModuleList([embedding_block for _ in range(signal_amount)])
        mixer_block = MHA_MixerBlock(signal_amount=signal_amount,
                                     sequence_length=SIGNAL_LENGTH // 2,
                                     embedding_size=64)
        self.Mixer = nn.Sequential(*[mixer_block for _ in range(layer_amount)])
        self.regression = nn.Sequential(
            nn.Flatten(1, 2),
            nn.AvgPool1d(kernel_size=64),
            nn.Flatten(),
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = x["signals"].unsqueeze(-1).permute(0, 1, 3, 2)
        # x shape: (batch_size, signal_amount, embedding_size, sequence_length)
        embedding_output = []
        for i in range(len(self.embedding)):
            embedding_output.append(self.embedding[i](x[:, i]))
        x = torch.stack(embedding_output, dim=1)
        del embedding_output
        x = x.permute(0, 1, 3, 2)
        # x shape: (batch_size, signal_amount, sequence_length, embedding_size)
        x = self.Mixer(x)
        x = self.regression(x)
        return x
"""
class IMSF_Net(nn.Module):
    # Should be renamed Di_CNN later on
    def __init__(self):
        super(IMSF_Net, self).__init__()
        blocks_setting = [
            # b1, b2, b3, b4
            #[16, 16, 16, 16],
            #[64, 64, 64, 64],
            # h, l, fu
            [6, 24, 18],
            [64, 64, 192],
        ]
        modules = []
        embedding_size = SIGNAL_LENGTH
        for i in range(len(blocks_setting)):
            #b1, b2, b3, b4 = blocks_setting[i]
            #modules.append(DiCNN_Block(b1, b2, b3, b4))
            h, l, fu = blocks_setting[i]
            modules.append(ModifiedIMSFBlock(l, h, fu))
            modules.append(nn.MaxPool1d(2))
            embedding_size //= 2
            modules.append(ECA(l + fu))
            #modules.append(ECA(b1 + b2 + b3 + b4))
        self.features = nn.Sequential(*modules)
        self.transformer = TransformerEncoder(
            head_amount=4,
            embedding_size=embedding_size,
            layer_amount=1,
            dropout=0.1
        )
        self.regression = nn.Sequential(
            nn.AvgPool1d(kernel_size=embedding_size),
            nn.Flatten(),
            nn.LazyLinear(2),
        )

    def forward(self, x):
        x = x["signals"]
        x = self.features(x)
        x = self.transformer(x)
        x = self.regression(x)
        return x
        
"""
"""
class IMSF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        blocks_setting = [
            # h, l, fu
            [6, 24, 18],
            [64, 64, 192],
        ]
        modules = []
        for i in range(len(blocks_setting)):
            h, l, fu = blocks_setting[i]
            modules.append(IMSF_Block(l, h, fu))
            modules.append(nn.MaxPool1d(2))
            modules.append(ECA(l + fu))
        self.IMSF = nn.Sequential(*modules)
        self.cnn = nn.Sequential(
            nn.LazyConv1d(256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(256, 512, 1)
        self.regression = nn.Sequential(
            #nn.AvgPool1d(kernel_size=256),
            #nn.Flatten(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = x["signals"]
        x = self.IMSF(x)
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.regression(x)
        return x
"""
class Inception_Transformer(nn.Module):
    def __init__(self):
        super(Inception_Transformer, self).__init__()
        self.conv1by1 = nn.Sequential(
            nn.LazyConv1d(64 * 1, kernel_size=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3by3 = nn.Sequential(
            nn.LazyConv1d(64 * 1, kernel_size=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            nn.LazyConv1d(64 * 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            #nn.LazyConv1d(64 * 3, kernel_size=1),
            #nn.BatchNorm1d(64 * 3),
            #nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv5by5 = nn.Sequential(
            nn.LazyConv1d(64 * 1, kernel_size=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            nn.LazyConv1d(64 * 1, kernel_size=5, padding=2),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            #nn.LazyConv1d(64 * 3, kernel_size=1),
            #nn.BatchNorm1d(64 * 3),
            #nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.pool3by3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv1d(64 * 1, kernel_size=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
        )
        self.conv1by1combine = nn.Sequential(
            nn.LazyConv1d(64 * 4, kernel_size=1),
            nn.BatchNorm1d(64 * 4),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.transformer = TransformerEncoder(
            head_amount=4,
            embedding_size=64 * 4,
            layer_amount=1,
            dropout=0.1
        )
        self.GlobalAvgPool = nn.AvgPool1d(kernel_size=64 * 4)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1by1(x)
        x2 = self.conv3by3(x)
        x3 = self.conv5by5(x)
        x4 = self.pool3by3(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv1by1combine(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.GlobalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)

        return x

class Dense_Transformer(nn.Module):
    def __init__(self):
        super(Dense_Transformer, self).__init__()
        self.embedding = nn.Sequential(
            nn.LazyLinear(1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(1024),
        )
        self.transformer = TransformerEncoder(
            head_amount=8,
            embedding_size=256,
            layer_amount=2,
            dropout=0.1
        )
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
            nn.ReLU()
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.view(x.size(0), -1, 256)
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)

        return x

class CNN_Transformer(nn.Module):
    def __init__(self):
        super(CNN_Transformer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.LazyConv1d(32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.transformer = TransformerEncoder(
            head_amount=8,
            embedding_size=SIGNAL_LENGTH // 4,
            layer_amount=2,
            dropout=0.1
        )
        self.GlobalAvgPool = nn.AvgPool1d(kernel_size=SIGNAL_LENGTH // 4)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transformer(x)
        x = self.GlobalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)

        return x

class CNN_Only(nn.Module):
    def __init__(self, bias_init=None):
        super(CNN_Only, self).__init__()
        cnn_blocks_setting = [
            # channel, reduction
            [32, 2],
            [40, 2],
            [64, 2],
            [80, 2],
            [160, 2]
        ]
        cnn_branch = []
        flatten_branch = []

        embedding_size = SIGNAL_LENGTH
        cnn_channel = len(SIGNALS_LIST)
        for i in range(len(cnn_blocks_setting)):
            out_channel, reduction = cnn_blocks_setting[i]
            embedding_size //= reduction
            cnn = [
                ESPResidual_Block(cnn_channel, out_channel),
                LCA(out_channel),
                nn.Conv1d(out_channel, out_channel, kernel_size=1),
                nn.BatchNorm1d(out_channel),
            ]
            if reduction != 1:
                cnn.insert(1, nn.MaxPool1d(reduction))
            cnn = nn.Sequential(*cnn)
            cnn_branch.append(cnn)
            cnn_channel = out_channel

        flatten_branch.append(nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1)
        ))

        self.cnn_branch = nn.ModuleList(cnn_branch)
        self.flatten = nn.ModuleList(flatten_branch)

        regression_head = nn.Linear(cnn_channel, 2)
        if bias_init is not None:
            with torch.no_grad():
                regression_head.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            regression_head,
            nn.ReLU()
        )

    def forward(self, x):
        cnn_output = x['signals']
        for i in range(len(self.cnn_branch)):
            cnn_output = self.cnn_branch[i](cnn_output)
        cnn_output = self.flatten[0](cnn_output)
        x = self.regression_head(cnn_output)
        return x

class Inception_Module(nn.Module):
    def __init__(self, out_channels, shrink_factor=4):
        super(Inception_Module, self).__init__()
        self.conv1by1 = nn.Sequential(
            nn.LazyConv1d(out_channels // shrink_factor, kernel_size=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
        )
        self.conv3by3 = nn.Sequential(
            nn.LazyConv1d(out_channels // shrink_factor, kernel_size=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
            nn.Conv1d(out_channels // shrink_factor, out_channels // shrink_factor, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
        )
        self.conv5by5 = nn.Sequential(
            nn.LazyConv1d(out_channels // shrink_factor, kernel_size=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
            nn.Conv1d(out_channels // shrink_factor, out_channels // shrink_factor, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
            nn.Conv1d(out_channels // shrink_factor, out_channels // shrink_factor, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
        )
        self.pool3by3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv1d(out_channels // shrink_factor, kernel_size=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
        )
        self.conv1by1combine = nn.Sequential(
            nn.Conv1d(out_channels // shrink_factor * 4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1by1(x)
        x2 = self.conv3by3(x)
        x3 = self.conv5by5(x)
        x4 = self.pool3by3(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv1by1combine(x)

        return x

'''
class Transformer_Only(nn.Module):
    def __init__(self, bias_init=None, alpha=0.5):
        super(Transformer_Only, self).__init__()
        self.alpha = alpha
        self.embedding = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.transformer = nn.Sequential(
            SimpleTransformerEncoder(
                sequence_length=SIGNAL_LENGTH // 2,
                head_amount=4,
                embedding_size=64,
                layer_amount=2,
                dropout=0.1
            ),
            nn.AvgPool1d(kernel_size=64),
            nn.Flatten(),
        )
        self.conv = nn.Sequential(
            #Inception_Module(128, shrink_factor=16),
            #nn.AvgPool1d(kernel_size=1024),
            #nn.Flatten(),
            #nn.LazyLinear(512),
            #nn.BatchNorm1d(512),
            MobileNetV2(output_size=512, input_size=1024, width_mult=1.0),
        )
        """
        self.demographics_embedding = nn.Sequential(
            #nn.LazyLinear(256),
            #nn.BatchNorm1d(256),
            #nn.ReLU(),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        """
        BP_regressor = nn.Linear(512,2)
        if bias_init is not None:
            with torch.no_grad():
                BP_regressor.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            BP_regressor,
            nn.ReLU()
        )

    def forward(self, x):
        #demographics = x['demographics']
        x_conv = self.conv(x['signals'])
        x = self.embedding(x['signals'])
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        #demographics = self.demographics_embedding(demographics)
        # Element-wise addition of x and demographics
        #x = x + demographics
        # Element-wise addition of x and conv scaled by alpha
        #  x = x * self.alpha + x_conv * (1 - self.alpha)# kinda bad
        x = x + x_conv
        x = self.regression_head(x)
        return x
'''
class Dummy_Model(nn.Module):
    def __init__(self):
        super(Dummy_Model, self).__init__()
        self.regresion_head = nn.Sequential(
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = x['signals']
        x = x.view(x.size(0), -1)
        x = self.regresion_head(x)
        return x

if __name__ == '__main__':
    model = Transformer_Only()
    x = {
        'signals': torch.FloatTensor(torch.randn(128, 3, SIGNAL_LENGTH)),
        # 'demographics': torch.FloatTensor(torch.randn(BATCH_SIZE, 3))
        # 'targets': torch.FloatTensor(torch.randn(128, 2))
    }
    """
    loss_fn = nn.L1Loss()
    for param in model.parameters():
        param.grad = None
    y_pred = model(x)  # Forward pass
    loss = loss_fn(y_pred, x['targets'])  # Compute the loss
    loss.backward()  # Backward pass

    gradient_norm_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_norm_dict[name] = param.grad.detach().norm().item()
    print(gradient_norm_dict)
    """
    with torch.no_grad():
        print(x)
        y = model(x)
        print(y)
        print(y.shape)
        torchinfo.summary(model, input_data=[x], verbose=1)

