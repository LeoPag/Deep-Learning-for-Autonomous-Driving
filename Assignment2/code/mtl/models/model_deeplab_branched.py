import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelDeepLabBranched(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_seg = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_dep = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_seg = DecoderDeeplabV3p(256, ch_out_encoder_4x, outputs_desc['semseg'])
        self.decoder_dep = DecoderDeeplabV3p(256, ch_out_encoder_4x, outputs_desc['depth'])

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])
        # print("!!!!!!ORIGINAL_INPUT!!!!!!", input_resolution)

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_tasks_seg = self.aspp_seg(features_lowest)
        features_tasks_dep = self.aspp_dep(features_lowest)
        # print('!!!!!OUT_ASSP_SEG!!!!!',features_tasks_seg.shape)
        # print('!!!!!OUT_ASSP_DEP!!!!!',features_tasks_dep.shape)

        predictions_4x_seg, _ = self.decoder_seg(features_tasks_seg, features[4])
        predictions_4x_dep, _ = self.decoder_dep(features_tasks_dep, features[4])
        # print('!!!!!OUT_DEC_SEG!!!!!', predictions_4x_seg.shape)
        # print('!!!!!OUT_DEC_DEP!!!!!', predictions_4x_dep.shape)

        predictions_1x_seg = F.interpolate(predictions_4x_seg, size=input_resolution, mode='bilinear', align_corners=False)
        predictions_1x_dep = F.interpolate(predictions_4x_dep, size=input_resolution, mode='bilinear', align_corners=False)
        # print('!!!!!PRED_SEG!!!!!', predictions_1x_seg.shape)
        # print('!!!!!PRED_DEP!!!!!', predictions_1x_dep.shape)

        out = {'semseg': predictions_1x_seg, 'depth': predictions_1x_dep}

        return out
