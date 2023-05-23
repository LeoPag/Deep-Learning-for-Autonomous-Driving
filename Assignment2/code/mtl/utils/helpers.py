from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from mtl.datasets.dataset_miniscapes import DatasetMiniscapes
from mtl.models.model_deeplab_v3_plus import ModelDeepLabV3Plus
from mtl.models.model_deeplab_branched import ModelDeepLabBranched
from mtl.models.model_deeplab_distilled import ModelDeepLabDistilled
from mtl.models.model_deeplab_distilled_se import ModelDeepLabDistilledSE
from mtl.models.model_deeplab_branched_se import ModelDeepLabBranchedSE


def resolve_dataset_class(name):
    return {
        'miniscapes': DatasetMiniscapes,
    }[name]


def resolve_model_class(name):
    return {
        'deeplabv3p': ModelDeepLabV3Plus,
        'deeplab_branched': ModelDeepLabBranched,
        'deeplab_distilled':ModelDeepLabDistilled,
        'deeplab_distilled_se': ModelDeepLabDistilledSE,
        'deeplab_branched_se': ModelDeepLabBranchedSE,
    }[name]


def resolve_optimizer(cfg, params):
    if cfg.optimizer == 'sgd':
        return SGD(
            params,
            lr=cfg.optimizer_lr,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == 'adam':
        return Adam(
            params,
            lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'poly':
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / cfg.num_epochs) ** cfg.lr_scheduler_power)
        )
    else:
        raise NotImplementedError
