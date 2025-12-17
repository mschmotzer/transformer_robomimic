import torch
import torch.nn as nn
import numpy as np
import textwrap
from torch import Tensor
from robomimic.models.obs_core import EncoderCore

class NormEncoder(EncoderCore):
    """
    Encoder core for flat (1-D) inputs that applies only Batch Normalization.
    Expected kwargs (passed via core_kwargs from config):
        feature_dimension (int, optional): number of features for batch norm.
    """
    def __init__(self, input_shape, feature_dimension=None):
        super(NormEncoder, self).__init__(input_shape=input_shape)  # <-- pass input_shape
        in_dim = int(np.prod(input_shape))
        out_dim = in_dim if feature_dimension is None else int(feature_dimension)
        self._bn =nn.LayerNorm(out_dim)  # BatchNorm for 1-D inputs
        self._output_dim = out_dim

    def output_shape(self, input_shape=None):
        return [self._output_dim]

    def forward(self, inputs: Tensor):
        # flatten inputs to [batch, features]
        x = inputs.view(inputs.size(0), -1)
        return self._bn(x)

    def __repr__(self):
        header = '{}'.format(str(self.__class__.__name__))
        msg = textwrap.indent("\noutput_dim={}\nbatch_norm={}".format(self._output_dim, self._bn), '  ')
        return header + '(' + msg + '\n)'
