import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, EncoderSE, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p, DecoderDeeplabV3p_noskip, SelfAttention


class ModelDeepLabDistilledSE(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc

        self.encoder = EncoderSE(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp_seg = ASPP(ch_out_encoder_bottleneck, 256)
        self.aspp_dep = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_1 = DecoderDeeplabV3p(256, ch_out_encoder_4x, outputs_desc['semseg'])
        self.decoder_2 = DecoderDeeplabV3p(256, ch_out_encoder_4x, outputs_desc['depth'])

        self.attention = SelfAttention(256, 256)

        self.decoder_3 = DecoderDeeplabV3p_noskip(256, outputs_desc['semseg'])
        self.decoder_4 = DecoderDeeplabV3p_noskip(256, outputs_desc['depth'])

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])
        # print("!!!!!!ORIGINAL_INPUT!!!!!!", input_resolution)

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        print('ARRIVOOOO',features_lowest.shape)

        features_tasks_seg = self.aspp_seg(features_lowest)
        features_tasks_dep = self.aspp_dep(features_lowest)
        # print('!!!!!OUT_ASSP_SEG!!!!!', features_tasks_seg.shape)
        # print('!!!!!OUT_ASSP_DEP!!!!!', features_tasks_dep.shape)

        predictions_4x_seg, features_4x_seg = self.decoder_1(features_tasks_seg, features[4])
        predictions_4x_dep, features_4x_dep = self.decoder_2(features_tasks_dep, features[4])
        # print('!!!!!OUT_DEC1!!!!!', predictions_4x_seg.shape)
        # print('!!!!!OUT_DEC2!!!!!', predictions_4x_dep.shape)
        # print('!!!!!FEAT_DEC_1!!!!!', features_4x_seg.shape)
        # print('!!!!!FEAT_DEC_2!!!!!', features_4x_dep.shape)

        predictions_1x_seg = F.interpolate(predictions_4x_seg, size=input_resolution, mode='bilinear', align_corners=False)
        predictions_1x_dep = F.interpolate(predictions_4x_dep, size=input_resolution, mode='bilinear', align_corners=False)
        # print('!!!!!PRED_SEG_1!!!!!', predictions_1x_seg.shape)
        # print('!!!!!PRED_DEP_1!!!!!', predictions_1x_dep.shape)

        features_4x_4 = self.attention(features_4x_seg)  # goes into decoder 4
        features_4x_3 = self.attention(features_4x_dep)  # goes into decoder 3
        # print('!!!!!POST_ATT_IN_4!!!!!', features_4x_4.shape)
        # print('!!!!!POST_ATT_IN_3!!!!!', features_4x_3.shape)

        predictions_4x_seg_2, _ = self.decoder_3(features_4x_3 + features_4x_seg, features[4])
        predictions_4x_dep_2, _ = self.decoder_4(features_4x_4 + features_4x_dep, features[4])
        # print('!!!!!OUT_DEC3!!!!!', predictions_4x_seg_2.shape)
        # print('!!!!!OUT_DEC4!!!!!', predictions_4x_dep_2.shape)

        predictions_1x_seg_2 = F.interpolate(predictions_4x_seg_2, size=input_resolution, mode='bilinear',
                                           align_corners=False)
        predictions_1x_dep_2 = F.interpolate(predictions_4x_dep_2, size=input_resolution, mode='bilinear',
                                           align_corners=False)
        # print('!!!!!PRED_SEG_2!!!!!', predictions_1x_seg_2.shape)
        # print('!!!!!PRED_DEP_2!!!!!', predictions_1x_dep_2.shape)

        out = {'semseg': [predictions_1x_seg, predictions_1x_seg_2], 'depth': [predictions_1x_dep, predictions_1x_dep_2]}

        return out
