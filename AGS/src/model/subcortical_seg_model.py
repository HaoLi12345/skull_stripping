import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from AGS.src.model.attention_gate import Attention_gate
"""
code modified from https://github.com/wolny/pytorch-3dunet/tree/master/pytorch3dunet/unet3d
03-27-2020, Hao
"""


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def conv3d(in_channels, out_channels, kernel_size, padding):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)


def create_conv(in_channels, out_channels, kernel_size, order, padding):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'r', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'c':
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, padding=padding)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'r', 'c']")

    return modules


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cbr', padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='cbr', padding=1):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels
            # conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order,
                                   padding=padding))


class ExtResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv,
                 conv_layer_order='cbr', padding=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        if basic_module == DoubleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                             encoder=True,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             padding=padding)
        else:
            self.basic_module = basic_module(in_channels, out_channels,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)

        return x


class Upsampling(nn.Module):

    def __init__(self, transposed_conv, in_channels=None, out_channels=None, kernel_size=3,
                 scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
            #                                    padding=1)
            # self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=scale_factor,
            #                                    padding=0)

            self.upsample = partial(self._interpolate, mode=mode)
            # self.upsample = F.interpolate(x, size=size, mode='nearest')
    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='cbr', padding=1):
        super(Decoder, self).__init__()
        # if basic_module == SingleConv:
        # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
        self.upsampling = Upsampling(transposed_conv=True, in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=conv_kernel_size, scale_factor=scale_factor)
        # concat joining
        self.joining = partial(self._joining, concat=True)
        if basic_module == DoubleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                             encoder=False,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             padding=padding)
        else:
            self.basic_module = basic_module(in_channels, out_channels,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             padding=padding)

        # self.conv = SingleConv(in_channels, out_channels, kernel_size=3, order='cbr', padding=1)
        # self.drop = nn.Dropout3d(p=0.1)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)

        # x = self.conv(x)
        # x = self.drop(x)

        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=2, pool_type='max', basic_module=DoubleConv,
                 conv_layer_order='cbr', padding=1):
        super(Bottleneck, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        if basic_module == DoubleConv:
            self.basic_module = basic_module(in_channels, out_channels,
                                             encoder=True,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             padding=padding)

        else:
            self.basic_module = basic_module(in_channels, out_channels,
                                             kernel_size=conv_kernel_size,
                                             order=conv_layer_order,
                                             padding=padding)




    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)

        x = self.basic_module(x)

        return x



class Middle_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, basic_module=SingleConv,
                 conv_layer_order='cbr', padding=1):

        super(Middle_Conv, self).__init__()

        self.basic_module = basic_module(in_channels, out_channels,
                                         kernel_size=5,
                                         order=conv_layer_order,
                                         padding=2)

        # self.basic_module3 = basic_module(in_channels, out_channels,
        #                                  kernel_size=3,
        #                                  order=conv_layer_order,
        #                                  padding=1)

        # self.basic_module1 = basic_module(in_channels, out_channels,
        #                                  kernel_size=1,
        #                                  order=conv_layer_order,
        #                                  padding=0)
        # self.relu = nn.ReLU()

    def forward(self, x):

        x5 = self.basic_module(x)
        # x3 = self.basic_module3(x)
        # x1 = self.basic_module1(x)

        # x = x1 + x3 + x5
        #
        # x = self.relu(x)

        x = x5

        return x

class Unet3d(nn.Module):
    def __init__(self, in_channels, out_channels, basic_module_encoder=DoubleConv, basic_module_decoder=DoubleConv,
                 f_maps=32, layer_order='cbr',
                 num_levels=5, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, classification=False):
        super(Unet3d, self).__init__()
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        self.classification = classification
        encoders = []
        encoders_second = []
        f_maps_short = f_maps[0:-1]
        # for i, out_feature_num in enumerate(f_maps_short):
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num,
                                  apply_pooling=False,  # skip pooling in the firs encoder
                                  basic_module=basic_module_encoder,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  padding=conv_padding)


            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  basic_module=basic_module_encoder,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding)


            encoders.append(encoder)


        self.encoders = nn.ModuleList(encoders)

        # self.bottleneck = Bottleneck(f_maps[-2], f_maps[-1],
        #                              basic_module=ExtResNetBlock,
        #                              conv_layer_order=layer_order,
        #                              conv_kernel_size=conv_kernel_size,
        #                              pool_kernel_size=pool_kernel_size,
        #                              padding=conv_padding
        #                              )


        if self.classification:
            self.fc1 = nn.Linear(f_maps[-1] * 6 * 6 * 3, 1000)
            self.fc2 = nn.Linear(1000, 100)
            self.fc3 = nn.Linear(100, 4)


        # a = self.encoders_second[0]
        # self.encoders_second[0].basic_module.SingleConv1.conv.in_channels = 10

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        attention_gates = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module_encoder == DoubleConv:
                # in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]

            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module_decoder,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              padding=conv_padding)
            decoders.append(decoder)

            # attention_gate = Attention_gate(out_feature_num, reversed_f_maps[i])
            # attention_gates.append(attention_gate)

        self.decoders = nn.ModuleList(decoders)
        # self.attention_gates = nn.ModuleList(attention_gates)

        # self.encoders_second = nn.ModuleList(encoders_second)
        # self.decoders_second = nn.ModuleList(decoders)
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels

        final_convs = []
        for i in range(len(reversed_f_maps) - 1):
            final_conv = nn.Conv3d(reversed_f_maps[i + 1], out_channels, 1)
            final_convs.append(final_conv)
        self.final_convs = nn.ModuleList(final_convs)

        self.final_conv = nn.Conv3d(f_maps[-1], out_channels, 1)

        #self.final_conv1 = nn.Conv3d(f_maps[0], out_channels, 1)
        final_convs2 = []
        a = [3,3,3,3]
        b = [1,1,1,1]

        for i in range(len(reversed_f_maps) - 1):
            final_conv2 = Middle_Conv(reversed_f_maps[i + 1], reversed_f_maps[i + 1], basic_module=SingleConv, conv_kernel_size=3, padding=1)
            final_convs2.append(final_conv2)
        self.final_convs2 = nn.ModuleList(final_convs2)



    def forward(self, x):
        encoders_output = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_output.insert(0, x)

        # x = self.bottleneck(x)
        # # remove the last encoder's output from the list
        # # !!remember: it's the 1st in the list
        encoders_output = encoders_output[1:]

        if self.classification:
            flat1 = encoders_output[-1].view(-1, self.num_flat_features(encoders_output[-1]))
            flat2 = encoders_output[-2].view(-1, self.num_flat_features(encoders_output[-2]))
            flat3 = encoders_output[-3].view(-1, self.num_flat_features(encoders_output[-3]))
            flat4 = encoders_output[-4].view(-1, self.num_flat_features(encoders_output[-4]))

            flat5 = x.view(-1, self.num_flat_features(x))
            flat = torch.cat((flat3, flat4, flat5), 1)
            x_class = self.fc1(flat5)
            x_class = self.fc2(x_class)
            x_class = self.fc3(x_class)

            # x_class = torch.sigmoid(x_class)



        # decoder part
        # for decoder, encoders_output in zip(self.decoders, encoders_output):
        #     # pass the output from the corresponding encoder and the output
        #     # of the previous decoder
        #     x = decoder(encoders_output, x)
        # x = self.final_conv1(x)

        # final_convs_output = []
        # old = self.final_conv(code)
        # for decoder, attention_gate, final_conv, encoder_output in zip(self.decoders, self.attention_gates, self.final_convs, encoders_output):
        #     # pass the output from the corresponding encoder and the output
        #     # of the previous decoder
        #     encoder_output, factor = attention_gate(encoder_output, x)
        #     x = decoder(encoder_output, x)
        #
        #     size = [encoder_output.size()[2], encoder_output.size()[3], encoder_output.size()[4]]
        #     old = F.interpolate(old, size=size)
        #     final_conv_output = final_conv(x)
        #     old = final_conv_output + old
        #
        # x = old


        old = self.final_conv(x)
        for decoder, final_conv, encoder_output, final_conv2 in zip(self.decoders, self.final_convs,
                                                                    encoders_output, self.final_convs2):

            # encoder_output1 = encoder_output + final_conv2(encoder_output)
            # x = decoder(encoder_output1, x)

            encoder_output = final_conv2(encoder_output)
            x = decoder(encoder_output, x)

            size = [encoder_output.size()[2], encoder_output.size()[3], encoder_output.size()[4]]

            old = F.interpolate(old, size=size)
            final_conv_output = final_conv(x)
            old = final_conv_output + old

        x = old







        # x = self.final_conv(x)

        # # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        # if self.testing and self.final_activation is not None:
        #     x = self.final_activation(x)
        if self.classification:
            return x, x_class
        else:
            return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




if __name__ == '__main__':
    model = Unet3d(1, 9).to('cuda')
    print(model)
    random_tensor = torch.rand([1, 1, 176, 176, 192]).to('cuda')
    output = model(random_tensor)
    print(output.shape)
