import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet


class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BottleneckSE(torch.nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckSE, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv1x1(inplanes, width)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = resnet.conv3x3(width, width, dilation=dilation, stride=stride, groups=groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = resnet.conv1x1(width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.se = SqueezeAndExcitation(planes*self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #print('pre_SE',out.shape)

        out = self.se(out)
        print('post_SE', out.shape)

        if self.downsample is not None:
            identity = self.downsample(x)

        print('downsample', self.downsample)

        print('identity', identity.shape)

        out += identity
        out = self.relu(out)

        print('\n')

        return out


_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}


def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    return ch_out_encoder_bottleneck, ch_out_encoder_4x


class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', True)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, True))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out


class EncoderSE(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', True)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BottleneckSE, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, True))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out

# Before task 1.3

"""
class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)

    def forward(self, features_bottleneck, features_skip_4x):

        # DeepLabV3+ style decoder
        # :param features_bottleneck: bottleneck features of scale > 4 (coming from ASPP)
        # :param features_skip_4x: features of encoder of scale == 4 (coming from solid blue block)
        # :return: features with 256 channels and the final tensor of predictions

        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        features_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        predictions_4x = self.features_to_predictions(features_4x)
        return predictions_4x, features_4x
"""

# After task 1.3


class DecoderDeeplabV3p_noskip(torch.nn.Module):
    def __init__(self, bottleneck_ch, num_out_ch):
        super(DecoderDeeplabV3p_noskip, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following
        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)

    def forward(self, features_bottleneck, features_skip_4x):

        # DeepLabV3+ style decoder
        # :param features_bottleneck: bottleneck features of scale > 4 (coming from ASPP)
        # :param features_skip_4x: features of encoder of scale == 4 (coming from solid blue block)
        # :return: features with 256 channels and the final tensor of predictions

        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        features_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        predictions_4x = self.features_to_predictions(features_4x)
        return predictions_4x, features_4x


class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        # TODO: Implement a proper decoder with skip connections instead of the following

        self.conv1_bn = torch.nn.Sequential(
            torch.nn.Conv2d(skip_4x_ch, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(),
        )

        self.conv3_bn = torch.nn.Sequential(
            torch.nn.Conv2d(48+bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU(),
            torch.nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            torch.nn.BatchNorm2d(bottleneck_ch),
            torch.nn.ReLU(),
        )

        self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)

        self.se = SqueezeAndExcitation(bottleneck_ch)

    def forward(self, features_bottleneck, features_skip_4x):
        
        # DeepLabV3+ style decoder
        # :param features_bottleneck: bottleneck features of scale > 4 (coming from ASPP)
        # :param features_skip_4x: features of encoder of scale == 4 (coming from solid blue block)
        # :return: features with 256 channels and the final tensor of predictions
        
        # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
        #       tensors in the same order and of the same shape.
        features_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
        )
        # print("!!!!!!LOW_LEVEL!!!!!!", features_skip_4x.shape)
        conv_skip = self.conv1_bn(features_skip_4x)
        # print("!!!!!!POST_CONV_SKIP!!!!!!", conv_skip.shape)
        features_4x = torch.cat((features_4x, conv_skip), dim=1)
        # print("!!!!!!SKIP_BOTT_STACK!!!!!!", features_4x.shape)
        features_4x = self.conv3_bn(features_4x)
        features_4x = self.se(features_4x)
        # print("!!!!!!FINAL_FEATURES!!!!!!", features_4x.shape)
        predictions_4x = self.features_to_predictions(features_4x)
        # print("!!!!!!PREDICTIONS_PRE_UP4!!!!!!", predictions_4x.shape)

        return predictions_4x, features_4x


class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

# Before task 1.3

"""
class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        # TODO: Implement ASPP properly instead of the following
        self.conv_out = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        # TODO: Implement ASPP properly instead of the following
        out = self.conv_out(x)
        return out
"""

# After task 1.3


class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        # TODO: Implement ASPP properly instead of the following
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.dil0_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0], bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.dil1_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1], bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.dil2_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2], bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.conv_bn_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        self.conv_bn_out_fin = torch.nn.Sequential(
            torch.nn.Conv2d(5*out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        self.se = SqueezeAndExcitation(out_channels)

    def forward(self, x):
        # TODO: Implement ASPP properly instead of the following
        input_res = x.shape[2:]
        conv_out = self.conv_out(x)
        dil0_out = self.dil0_out(x)
        dil1_out = self.dil1_out(x)
        dil2_out = self.dil2_out(x)
        avgpool = torch.nn.AvgPool2d(input_res)
        img_pool_out = avgpool(x)
        img_pool_out = self.conv_bn_out(img_pool_out)
        upsample = torch.nn.Upsample(size=input_res, mode='bilinear')
        img_pool_out = upsample(img_pool_out)
        out = torch.cat((conv_out, dil0_out, dil1_out, dil2_out, img_pool_out), dim=1)
        out = self.conv_bn_out_fin(out)
        out = self.se(out)
        return out


class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed
