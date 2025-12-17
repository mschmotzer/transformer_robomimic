from robomimic.models.obs_core import EncoderCore
import abc
import numpy as np
import textwrap
import random

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import Lambda, Compose
import torchvision.transforms.functional as TVF

import robomimic.models.base_nets as BaseNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict

# NOTE: this is required for the backbone classes to be found by the `eval` call in the core networks
from robomimic.models.base_nets import *
from robomimic.utils.vis_utils import visualize_image_randomizer
from robomimic.macros import VISUALIZE_RANDOMIZER


class MLPEncoder(EncoderCore):
    """
    Encoder core for flat (1-D) inputs. Applies an MLP to the flattened input
    and optionally projects to a desired `feature_dimension`.
    Expected kwargs (passed via core_kwargs from config):
        layer_dims (list): hidden layer sizes for the MLP (optional)
        feature_dimension (int): output dimension after projection (optional)
        activation (nn.Module or str): activation to use between layers (defaults to ReLU)
    """
    def __init__(self, input_shape, layer_dims=None, output_dim=None, activation="relu"):
        super(MLPEncoder, self).__init__(input_shape=input_shape)
        in_dim = int(np.prod(input_shape))
        # default activation
        act_cls = None
        if activation is None:
            act_cls = None
        else:
            if isinstance(activation, str):
                if activation.lower() == "tanh":
                    act_cls = BaseNets.nn.Tanh
                else:
                    # fallback to ReLU for unknown strings
                    act_cls = BaseNets.nn.ReLU
            else:
                act_cls = activation
        layer_dims = [] if layer_dims is None else list(layer_dims)
        out_dim = in_dim if output_dim is None else int(output_dim)
        # create MLP that maps in_dim -> out_dim
        self._mlp = BaseNets.MLP(
            input_dim=in_dim,
            output_dim=out_dim,
            layer_dims=layer_dims,
            activation=act_cls if act_cls is not None else BaseNets.nn.ReLU,
        )
    def output_shape(self, input_shape=None):
        return [self._mlp._output_dim]
    def forward(self, inputs):
        # inputs may already be flat or have extra dims; flatten all but batch
        x = TensorUtils.flatten(inputs, begin_axis=1)
        return self._mlp(x)
    def __repr__(self):
        header = '{}'.format(str(self.__class__.__name__))
        msg = textwrap.indent("\ninput_shape={}\nmlp={}".format(self.input_shape, self._mlp), '  ')
        return header + '(' + msg + '\n)'
    
class MLPEncoderOutputBN(EncoderCore):
    """
    MLP encoder for flat (1-D) inputs where the output is normalized.
    Expected kwargs:
        layer_dims (list): hidden layer sizes for the MLP (optional)
        output_dim (int): output dimension after projection (optional)
        activation (nn.Module or str): activation to use (defaults to ReLU)
        norm_type (str): 'batch' or 'layer' normalization (default: 'batch')
    """
    def __init__(self, input_shape, layer_dims=None, output_dim=None, activation="relu", norm_type="batch"):
        super(MLPEncoderOutputBN, self).__init__(input_shape=input_shape)
        in_dim = int(np.prod(input_shape))

        # Determine activation
        if activation is None:
            act_cls = None
        elif isinstance(activation, str):
            if activation.lower() == "tanh":
                act_cls = BaseNets.nn.Tanh
            else:
                act_cls = BaseNets.nn.ReLU
        else:
            act_cls = activation

        layer_dims = [] if layer_dims is None else list(layer_dims)
        out_dim = in_dim if output_dim is None else int(output_dim)

        # Build MLP
        self._mlp = BaseNets.MLP(
            input_dim=in_dim,
            output_dim=out_dim,
            layer_dims=layer_dims,
            activation=act_cls if act_cls is not None else BaseNets.nn.ReLU,
        )

        # Normalization on output
        if norm_type.lower() == "batch":
            self._norm = nn.BatchNorm1d(out_dim)
        elif norm_type.lower() == "layer":
            self._norm = nn.LayerNorm(out_dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self._output_dim = out_dim

    def output_shape(self, input_shape=None):
        return [self._output_dim]

    def forward(self, inputs: Tensor):
        # Flatten inputs except batch
        x = TensorUtils.flatten(inputs, begin_axis=1)
        x = self._mlp(x)
        x = self._norm(x)  # normalize output
        return x

    def __repr__(self):
        header = '{}'.format(str(self.__class__.__name__))
        msg = textwrap.indent(
            "\ninput_shape={}\nmlp={}\noutput_norm={}".format(self.input_shape, self._mlp, self._norm),
            '  '
        )
        return header + '(' + msg + '\n)'
