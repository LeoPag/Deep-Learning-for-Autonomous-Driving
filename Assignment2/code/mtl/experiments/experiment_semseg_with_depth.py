import os
import pytorch_lightning as pl
import torch
import wandb
import copy
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

from mtl.datasets.definitions import *
from mtl.losses.loss_regression import LossRegression
from mtl.utils.metrics import MetricsSemseg, MetricsDepth
from mtl.utils.helpers import resolve_optimizer, resolve_dataset_class, resolve_model_class, resolve_lr_scheduler
from mtl.utils.transforms import get_transforms
from mtl.utils.transforms import get_transforms_mm
from mtl.utils.visualization import compose


class ExperimentSemsegDepth(pl.LightningModule):
    def __init__(self, cfg):
        super(ExperimentSemsegDepth, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters(self.cfg)

        dataset_class = resolve_dataset_class(cfg.dataset)
        self.datasets = {
            split: dataset_class(cfg.dataset_root, split, integrity_check=False)
            for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)
        }

        for split in (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST):
            print(f'Number of samples in {split} split: {len(self.datasets[split])}')

        self.semseg_num_classes = self.datasets[SPLIT_TRAIN].semseg_num_classes
        self.semseg_ignore_label = self.datasets[SPLIT_TRAIN].semseg_ignore_label
        self.semseg_class_names = self.datasets[SPLIT_TRAIN].semseg_class_names
        self.semseg_class_colors = self.datasets[SPLIT_TRAIN].semseg_class_colors
        self.rgb_mean = self.datasets[SPLIT_TRAIN].rgb_mean
        self.rgb_stddev = self.datasets[SPLIT_TRAIN].rgb_stddev
        self.depth_meters_mean = self.datasets[SPLIT_TRAIN].depth_meters_mean
        self.depth_meters_stddev = self.datasets[SPLIT_TRAIN].depth_meters_stddev
        self.depth_meters_min = self.datasets[SPLIT_TRAIN].depth_meters_min
        self.depth_meters_max = self.datasets[SPLIT_TRAIN].depth_meters_max

        outputs_descriptor = {
            MOD_SEMSEG: self.semseg_num_classes,
            MOD_DEPTH: 1,
        }

        model_class = resolve_model_class(cfg.model_name)
        self.net = model_class(cfg, outputs_descriptor)
        print(self.net)
        print('PARAMS', sum(p.numel() for p in self.net.parameters() if p.requires_grad))

        self.datasets[SPLIT_TRAIN].set_transforms(get_transforms(
            semseg_ignore_label=self.semseg_ignore_label,
            geom_scale_min=cfg.aug_geom_scale_min,
            geom_scale_max=cfg.aug_geom_scale_max,
            geom_tilt_max_deg=cfg.aug_geom_tilt_max_deg,
            geom_wiggle_max_ratio=cfg.aug_geom_wiggle_max_ratio,
            geom_reflect=cfg.aug_geom_reflect,
            crop_random=cfg.aug_input_crop_size,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            depth_meters_mean=self.depth_meters_mean,
            depth_meters_stddev=self.depth_meters_stddev,
            #depth_meters_max=self.depth_meters_max,
            #depth_meters_min=self.depth_meters_min  # Remove this and the previous line when you go back to get_transform
        ))
        self.transforms_val_test = get_transforms(
            semseg_ignore_label=self.semseg_ignore_label,
            crop_for_passable=32,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            depth_meters_mean=self.depth_meters_mean,
            depth_meters_stddev=self.depth_meters_stddev,
            #depth_meters_max=self.depth_meters_max,
            #depth_meters_min=self.depth_meters_min  # Remove this and the previous line when you go back to get_transform
        )
        self.datasets[SPLIT_VALID].set_transforms(self.transforms_val_test)
        self.datasets[SPLIT_TEST].set_transforms(self.transforms_val_test)

        self.loss_semseg = torch.nn.CrossEntropyLoss(ignore_index=self.semseg_ignore_label)
        self.loss_depth = LossRegression()

        self.metrics_semseg = MetricsSemseg(self.semseg_num_classes, self.semseg_ignore_label, self.semseg_class_names)
        self.metrics_depth = MetricsDepth()

    def training_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y_semseg_lbl = batch[MOD_SEMSEG].squeeze(1)
        y_depth = batch[MOD_DEPTH].squeeze(1)

        if torch.cuda.is_available():
            rgb = rgb.cuda()
            y_semseg_lbl = y_semseg_lbl.cuda()
            y_depth = y_depth.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat[MOD_SEMSEG]
        y_hat_depth = y_hat[MOD_DEPTH]

        if type(y_hat_semseg) is list:
            # deep supervision scenario: penalize all predicitons in the list and average losses
            loss_semseg = sum([self.loss_semseg(y_hat_semseg_i, y_semseg_lbl) for y_hat_semseg_i in y_hat_semseg])
            loss_depth = sum([self.loss_depth(y_hat_depth_i, y_depth) for y_hat_depth_i in y_hat_depth])
            loss_semseg = loss_semseg / len(y_hat_semseg)
            loss_depth = loss_depth / len(y_hat_depth)
            y_hat_semseg = y_hat_semseg[-1]
            y_hat_depth = y_hat_depth[-1]
        else:
            loss_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)
            loss_depth = self.loss_depth(y_hat_depth, y_depth)

        loss_total = self.cfg.loss_weight_semseg * loss_semseg + self.cfg.loss_weight_depth * loss_depth

        self.log_dict({
                'loss_train/semseg': loss_semseg,
                'loss_train/depth': loss_depth,
                'loss_train/total': loss_total,
            }, on_step=True, on_epoch=False, prog_bar=True
        )

        if self.can_visualize():
            self.visualize(batch, y_hat_semseg, y_hat_depth, batch[MOD_ID], 'imgs_train/batch_crops')

        return {
            'loss': loss_total,
        }

    def inference_step(self, batch):
        rgb = batch[MOD_RGB]

        if torch.cuda.is_available():
            rgb = rgb.cuda()

        y_hat = self.net(rgb)
        y_hat_semseg = y_hat[MOD_SEMSEG]
        y_hat_depth = y_hat[MOD_DEPTH]

        if type(y_hat_semseg) is list:
            y_hat_semseg = y_hat_semseg[-1]
            y_hat_depth = y_hat_depth[-1]

        y_hat_semseg_lbl = y_hat_semseg.argmax(dim=1)

        y_hat_depth_meters = (
                y_hat_depth * self.depth_meters_stddev + self.depth_meters_mean
        ).clamp(self.depth_meters_min, self.depth_meters_max)
        """
        y_hat_depth_meters = (
                torch.pow((self.depth_meters_max - self.depth_meters_min + 1), y_hat_depth) + self.depth_meters_min - 1
        ).clamp(self.depth_meters_min, self.depth_meters_max)
        """
        return y_hat_semseg, y_hat_semseg_lbl, y_hat_depth, y_hat_depth_meters

    def validation_step(self, batch, batch_nb):
        y_hat_semseg, y_hat_semseg_lbl, y_hat_depth, y_hat_depth_meters = self.inference_step(batch)

        y_semseg_lbl = batch[MOD_SEMSEG].squeeze(1)
        y_depth = batch[MOD_DEPTH].squeeze(1)

        if torch.cuda.is_available():
            y_semseg_lbl = y_semseg_lbl.cuda()
            y_depth = y_depth.cuda()

        y_depth_meters = y_depth * self.depth_meters_stddev + self.depth_meters_mean
        #y_depth_meters = torch.pow((self.depth_meters_max - self.depth_meters_min + 1), y_depth) + self.depth_meters_min - 1

        loss_val_semseg = self.loss_semseg(y_hat_semseg, y_semseg_lbl)
        loss_val_depth = self.loss_depth(y_hat_depth, y_depth)
        loss_val_total = loss_val_semseg + loss_val_depth

        self.metrics_semseg.update_batch(y_hat_semseg_lbl, y_semseg_lbl)
        self.metrics_depth.update_batch(y_hat_depth_meters, y_depth_meters)

        self.log_dict({
                'loss_val/semseg': loss_val_semseg,
                'loss_val/depth': loss_val_depth,
                'loss_val/total': loss_val_total,
            }, on_step=False, on_epoch=True
        )

    def validation_epoch_end(self, outputs):
        self.observer_step()

        metrics_semseg = self.metrics_semseg.get_metrics_summary()
        self.metrics_semseg.reset()

        metrics_depth = self.metrics_depth.get_metrics_summary()
        self.metrics_depth.reset()

        metric_semseg = metrics_semseg['mean_iou']
        metric_depth = metrics_depth['si_log_rmse']
        metric_multitask = (metric_semseg - 50).clamp(min=0) + (50 - metric_depth).clamp(min=0)

        scalar_logs = {
            'metrics_summary/semseg': metric_semseg,
            'metrics_summary/depth': metric_depth,
            'metrics_summary/multitask': metric_multitask,
            'trainer/LR': torch.tensor(self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0]),
        }
        scalar_logs.update({f'metrics_task_semseg/{k.replace(" ", "_")}': v for k, v in metrics_semseg.items()})
        scalar_logs.update({f'metrics_task_depth/{k}': v for k, v in metrics_depth.items()})

        self.log_dict(scalar_logs, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        _, y_hat_semseg_lbl, _, y_hat_depth_meters = self.inference_step(batch)  # WITHOUT AUGMENTATION
        #y_hat_semseg_lbl_tens = y_hat_semseg_lbl.unsqueeze(1).unsqueeze(4)
        #y_hat_semseg_lbl_tens = copy.copy(y_hat_semseg_lbl)
        #y_hat_semseg_lbl_tens = y_hat_semseg_lbl_tens.cuda()
        #y_hat_semseg_lbl_tens = y_hat_semseg_lbl_tens.unsqueeze(1).unsqueeze(4)


        #print('\n',"SHAPE1", y_hat_depth_meters.shape)

        scales = [0.5, 2, 3] #, 4, 5]
        #y_hat_semseg_lbl_list = [y_hat_semseg_lbl]
        #y_hat_depth_meters_list = [y_hat_depth_meters]

        #batch_int = copy.copy(batch)
        batch_int = {'id': batch['id']}

        for scale in scales:
            #print("scale", scale)
            batch_int[MOD_RGB] = F.interpolate(batch[MOD_RGB], scale_factor=scale, mode='bilinear')
            _, _, _, y_hat_depth_meters_int = self.inference_step(batch_int)
            #print("SHAPE2_post_inference", y_hat_depth_meters_int.shape)
            """
            y_hat_semseg_lbl_int = y_hat_semseg_lbl_int.unsqueeze(1)
            y_hat_semseg_lbl_int = y_hat_semseg_lbl_int.type(torch.cuda.FloatTensor)
            y_hat_semseg_lbl_int = F.interpolate(y_hat_semseg_lbl_int, scale_factor=1/scale, mode='bilinear')
            """
            y_hat_depth_meters_int = F.interpolate(y_hat_depth_meters_int, scale_factor=1/scale, mode='bilinear')
            """
            y_hat_semseg_lbl_int = y_hat_semseg_lbl_int.type(torch.cuda.LongTensor)
            y_hat_semseg_lbl_int = y_hat_semseg_lbl_int.unsqueeze(4)
            """
            #print("SHAPE2_post_reinterp", y_hat_depth_meters_int.shape)
            y_hat_depth_meters += y_hat_depth_meters_int
            """
            print("stackone", y_hat_semseg_lbl_tens.shape)
            print("new semseg", y_hat_semseg_lbl_int.shape)
            y_hat_semseg_lbl_tens = torch.cat((y_hat_semseg_lbl_tens, y_hat_semseg_lbl_int),dim=4)
            #y_hat_semseg_lbl_int = y_hat_semseg_lbl_int.cpu()
            #y_hat_semseg_lbl_list.append(y_hat_semseg_lbl_int)
            #y_hat_depth_meters_list.append(y_hat_depth_meters_int)
            """

        y_hat_depth_meters = y_hat_depth_meters / (len(scales) + 1)
        """
        y_hat_semseg_lbl_tens = y_hat_semseg_lbl_tens.squeeze(1)
        y_hat_semseg_lbl = torch.mode(y_hat_semseg_lbl_tens, dim=-1)
        y_hat_semseg_lbl = y_hat_semseg_lbl[0]
        #y_hat_semseg_lbl = y_hat_semseg_lbl.cuda()

        print("y hat semseg shape", y_hat_semseg_lbl.shape)
        """
        #print("y hat depth shape", y_hat_depth_meters.shape)

        path_pred = os.path.join(self.cfg.log_dir, 'predictions')
        path_pred_semseg = os.path.join(path_pred, MOD_SEMSEG)
        path_pred_depth = os.path.join(path_pred, MOD_DEPTH)
        if batch_nb == 0:
            os.makedirs(path_pred_semseg)
            os.makedirs(path_pred_depth)
        split_test = SPLIT_TEST
        for i in range(y_hat_semseg_lbl.shape[0]):
            sample_name = self.datasets[split_test].name_from_index(batch[MOD_ID][i])
            path_file_semseg = os.path.join(path_pred_semseg, f'{sample_name}.png')
            path_file_depth = os.path.join(path_pred_depth, f'{sample_name}.png')
            pred_semseg = y_hat_semseg_lbl[i]
            pred_depth = y_hat_depth_meters[i]
            self.datasets[split_test].save_semseg(
                path_file_semseg, pred_semseg, self.semseg_class_colors, self.semseg_ignore_label
            )
            self.datasets[split_test].save_depth(
                path_file_depth, pred_depth, out_of_range_policy='clamp_to_range'
            )

    def test_end(self, outputs):
        return {}

    def configure_optimizers(self):
        optimizer = resolve_optimizer(self.cfg, self.parameters())
        lr_scheduler = resolve_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return DataLoader(
            self.datasets[SPLIT_TRAIN],
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=True,
        )

    def create_val_test_dataloader(self, split):
        return DataLoader(
            self.datasets[split],
            self.cfg.batch_size_validation,
            shuffle=False,
            num_workers=self.cfg.workers_validation,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return self.create_val_test_dataloader(SPLIT_VALID)

    def test_dataloader(self):
        return self.create_val_test_dataloader(SPLIT_TEST)

    def visualize(self, batch, y_hat_semseg, y_hat_depth, rgb_tags, tag):
        batch = {k: v.cpu().detach() for k, v in batch.items() if torch.is_tensor(v)}
        y_hat_semseg_lbl = y_hat_semseg.cpu().detach().argmax(dim=1)
        y_hat_depth = y_hat_depth.cpu().detach()
        visualization_plan = [
            (MOD_RGB, batch[MOD_RGB], rgb_tags),
            (MOD_SEMSEG, batch[MOD_SEMSEG], 'GT SemSeg'),
            (MOD_SEMSEG, y_hat_semseg_lbl, 'Prediction SemSeg'),
            (MOD_DEPTH, batch[MOD_DEPTH], 'GT Depth'),
            (MOD_DEPTH, y_hat_depth, 'Prediction Depth'),
        ]
        vis = compose(
            visualization_plan,
            self.cfg,
            rgb_mean=self.rgb_mean,
            rgb_stddev=self.rgb_stddev,
            semseg_color_map=self.semseg_class_colors,
            semseg_ignore_label=self.semseg_ignore_label,
        )
        # Use commit=False to not increment the step counter
        self.logger.experiment[0].log({
            tag: [wandb.Image(vis.cpu(), caption=tag)]
        }, commit=False)

        # Tensorboard logging
        self.logger.experiment[2].add_image(tag, vis, self.global_step)

    def visualize_histograms(self, batch, y_hat_depth):
        batch = {k: v.cpu().detach() for k, v in batch.items() if torch.is_tensor(v)}
        y_hat_depth = y_hat_depth.cpu().detach()

        # visualize depth histograms to see how far the distribution of values is away from gaussian
        depth_normalized = batch[MOD_DEPTH]
        depth_normalized = depth_normalized[depth_normalized == depth_normalized]
        depth_meters = depth_normalized * self.depth_meters_stddev + self.depth_meters_mean
        y_hat_depth_meters = y_hat_depth * self.depth_meters_stddev + self.depth_meters_mean

        # Only lowercase letters work for the log names
        # Use commit=False to not increment the step counter
        self.logger.experiment[0].log({
            'histograms/gt_depth_normalized': wandb.Histogram(depth_normalized, num_bins=64),
            'histograms/gt_depth_meters': wandb.Histogram(depth_meters, num_bins=64),
            'histograms/pred_depth_normalized': wandb.Histogram(y_hat_depth, num_bins=64),
            'histograms/pred_depth_meters': wandb.Histogram(y_hat_depth_meters, num_bins=64),
        }, commit=False)

        # Tensorboard logging
        self.logger.experiment[2].add_histogram('GT_Depth_Normalized', depth_normalized, self.trainer.global_step, bins=64)
        self.logger.experiment[2].add_histogram('GT_Depth_Meters', depth_meters, self.trainer.global_step, bins=64)
        self.logger.experiment[2].add_histogram('Pred_Depth_Normalized', y_hat_depth, self.trainer.global_step, bins=64)
        self.logger.experiment[2].add_histogram('Pred_Depth_Meters', y_hat_depth_meters, self.trainer.global_step, bins=64)

    def can_visualize(self):
        return (not torch.cuda.is_available() or torch.cuda.current_device() == 0) and (
                self.global_step - self.cfg.num_steps_visualization_first) % \
                self.cfg.num_steps_visualization_interval == 0

    def observer_step(self):
        if torch.cuda.is_available() and torch.cuda.current_device() != 0:
            return
        vis_transforms = self.transforms_val_test
        list_samples = []
        for i in self.cfg.observe_train_ids:
            list_samples.append(self.datasets[SPLIT_TRAIN].get(i, override_transforms=vis_transforms))
        for i in self.cfg.observe_valid_ids:
            list_samples.append(self.datasets[SPLIT_VALID].get(i, override_transforms=vis_transforms))
        list_prefix = ('imgs_train/', ) * len(self.cfg.observe_train_ids) + ('imgs_val/', ) * len(self.cfg.observe_valid_ids)
        batch = default_collate(list_samples)
        rgb = batch[MOD_RGB]
        rgb_tags = [f'{prefix}{id}' for prefix, id in zip(list_prefix, batch[MOD_ID])]
        with torch.no_grad():
            if torch.cuda.is_available():
                rgb = rgb.cuda()
            y_hat = self.net(rgb)
            y_hat_semseg = y_hat[MOD_SEMSEG]
            y_hat_depth = y_hat[MOD_DEPTH]
            if type(y_hat_semseg) is list:
                y_hat_semseg = y_hat_semseg[-1]
                y_hat_depth = y_hat_depth[-1]
        self.visualize(batch, y_hat_semseg, y_hat_depth, rgb_tags, 'imgs_val/observed_samples')
        self.visualize_histograms(batch, y_hat_depth)
