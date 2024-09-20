import torch
import torch.nn as nn
import torchinfo
from dotenv import load_dotenv
import os

from Models.FNet import *
from Models.submodules import *
from Models.transformer import *

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

SIGNALS_LIST = [signal.strip().lower() for signal in os.getenv('SIGNALS').split(',')]
SIGNAL_LENGTH = int(os.getenv('INPUT_LENGTH'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))


class TransFMSD(nn.Module): # Benchmark Model
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


class MSCA_FNet_Benchmark(nn.Module):
    def __init__(self, bias_init=None):
        super(MSCA_FNet_Benchmark, self).__init__()
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
            [3, 2, 1, 3],   # 512 out   # 625
            [3, 2, 1, 32],                  # 256 out   # 312
            [3, 2, 1, 64],                  # 128_out   # 156
            #[3, 2, 1, 96],
        ]
        transformer_blocks_setting = [
            # patch_amount, reduction, head_amount, mlp_hidden
            [156, 1, 4, 512],
            [156, 2, 4, 512],
            [78, 1, 4, 512],
            [78, 1, 4, 512],
            [78, 1, 4, 512],
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
                #transformer.insert(0, LocalityInvariantShifting([i * (SIGNAL_LENGTH // shift_amount)
                #                                                 for i in range(shift_amount)]))
                transformer.insert(0, ConvPatchify_Block(patch_blocks_setting, embedding_size))
                #transformer.insert(2, PositionalEncoding(embedding_size))
                #transformer.insert(2, RandomEncoding(patch_amount, embedding_size))
            else:
                transformer.insert(0, Rearrange('batch embedding patch -> batch patch embedding'))
            if reduction != 1:
                transformer.append(SequencePooling1D(reduction))
            transformer.append(Rearrange('batch patch embedding -> batch embedding patch'))
            transformer = nn.Sequential(*transformer)
            transformer_branch.append(transformer)

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
                    # ChannelShuffle1D(2),
                    # GatedConv_Block(out_channel),
                    # nn.BatchNorm1d(out_channel),
                    # nn.Sequential(
                    #    nn.Conv1d(out_channel * 2, out_channel, kernel_size=1),
                    #    nn.BatchNorm1d(out_channel),
                    # )

                    # Excitation_Block(out_channel, embedding_size),
                    # Excitation_Block(embedding_size, out_channel),
                    # nn.Sequential(
                    #    nn.LazyLinear(128),
                    #    nn.Dropout(0.1),
                    #    nn.ReLU()
                    # ),

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
        self.feature_dim = 512
        self.feautre_head = nn.Sequential(
            nn.LazyLinear(512),
            nn.Dropout(0.1),
        )

        regression_head = nn.Linear(self.feature_dim, 2)
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
            transformer_output = self.transformer_branch[i](transformer_output)
            if i < len(self.fusion_branch) - 1:
                # temp, t_excitation = self.fusion_branch[i][0](cnn_output, transformer_output)
                # cnn_output, c_excitation = self.fusion_branch[i][1](transformer_output, cnn_output)
                temp = transformer_output + self.fusion_branch[i][0](cnn_output)
                cnn_output = cnn_output + self.fusion_branch[i][1](transformer_output)
                transformer_output = temp
                # t_excitation = self.fusion_branch[i][2](t_excitation)
                # c_excitation = self.fusion_branch[i][2](c_excitation)
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
        output = self.feautre_head(x)
        x = self.regression_head(output)
        return x, output

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
        self.feature_dim = cnn_channel
        regression_head = nn.Linear(self.feature_dim, 2)
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
        return x, cnn_output

class Dummy_Model(nn.Module):
    def __init__(self, bias_init=None):
        super(Dummy_Model, self).__init__()
        regression_head = nn.Linear(512, 2)
        if bias_init is not None:
            with torch.no_grad():
                regression_head.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            regression_head,
        )

    def forward(self, x):
        x = x['signals']
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)
        return x



if __name__ == '__main__':
    model = IMSF_Net()
    x = {
        'signals': torch.FloatTensor(torch.randn(100, 3, SIGNAL_LENGTH)),
        # 'demographics': torch.FloatTensor(torch.randn(BATCH_SIZE, 3))
        # 'targets': torch.FloatTensor(torch.randn(128, 2))
    }

    with torch.no_grad():
        print(x)
        y = model(x)
        print(y)
        print(y.shape)
        torchinfo.summary(model, input_data=[x], verbose=1)

