import torch
from torch import nn
from copy import deepcopy


class EMA(nn.Module):
    """ Model Exponential Moving Average V2
        https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/utils/model_ema.py
        decay=0.9999 means that when updating the model weights, we keep 99.99% of the previous model weights and only update 0.01% of the new weights at each iteration.
        ema_model_weights = decay * ema_model_weights + (1 - decay) * model_weights

        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        A smoothed version of the weights is necessary for some training schemes to perform well.
        E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
        RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
        smoothing of weights to match results. Pay attention to the decay constant you are using
        relative to your update count per epoch.
        """

    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.update_fn = lambda e, m: self.decay * e + (1. - self.decay) * m

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema_model.state_dict().values(),
                                      model.state_dict().values()):
                assert ema_v.shape == model_v.shape, 'wrong ema model!'
                ema_v.copy_(self.update_fn(ema_v, model_v))