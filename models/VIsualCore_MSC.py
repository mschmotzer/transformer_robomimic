import abc
import numpy as np
import textwrap
import random

import torch
import torch.nn as nn
from torchvision.transforms import Lambda, Compose
import torchvision.transforms.functional as TVF

import robomimic.models.base_nets as BaseNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.obs_core as EncoderCore
from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict

# NOTE: this is required for the backbone classes to be found by the `eval` call in the core networks
from robomimic.models.base_nets import *
from robomimic.utils.vis_utils import visualize_image_randomizer
from robomimic.macros import VISUALIZE_RANDOMIZER


class VisualCore_MSC(EncoderCore.VisualCore):
    """
    A network block that combines a visual backbone network with optional pooling
    and linear layers, followed by an optional MLP projection.
    """
    def __init__(
        self,
        input_shape,
        backbone_class="ResNet18Conv",
        pool_class="SpatialSoftmax",
        backbone_kwargs=None,
        pool_kwargs=None,
        flatten=True,
        feature_dimension=64,
        mlp_layer_dims=None,
        mlp_output_dim=None,
        mlp_activation="relu",
    ):
        """
        Args:
            input_shape (tuple): shape of input (not including batch dimension)
            backbone_class (str): class name for the visual backbone network. Defaults
                to "ResNet18Conv".
            pool_class (str): class name for the visual feature pooler (optional)
                Common options are "SpatialSoftmax" and "SpatialMeanPool". Defaults to
                "SpatialSoftmax".
            backbone_kwargs (dict): kwargs for the visual backbone network (optional)
            pool_kwargs (dict): kwargs for the visual feature pooler (optional)
            flatten (bool): whether to flatten the visual features
            feature_dimension (int): if not None, add a Linear layer to
                project output into a desired feature dimension
            mlp_layer_dims (list): dimensions of hidden layers for the MLP (optional)
            mlp_output_dim (int): output dimension of the MLP (optional)
            mlp_activation (str or callable): activation function for MLP. Options: "tanh", "relu", or a torch.nn activation class
        """
        print("SHPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP______________________", backbone_class)
        # Call parent __init__ with all the same parameters
        super(VisualCore_MSC, self).__init__(
            input_shape=input_shape,
            backbone_class=backbone_class,
            pool_class=pool_class,
            backbone_kwargs=backbone_kwargs,
            pool_kwargs=pool_kwargs,
            flatten=flatten,
            feature_dimension=feature_dimension,
        )
        
        # Get the output shape from the parent's visual core
        visual_output_shape = super(VisualCore_MSC, self).output_shape(input_shape)
        print("VisualCore_MSC visual output shape:", visual_output_shape)
        in_dim = int(np.prod(visual_output_shape))
        
        # Setup MLP activation
        act_cls = None
        if mlp_activation is None:
            act_cls = None
        else:
            if isinstance(mlp_activation, str):
                if mlp_activation.lower() == "tanh":
                    act_cls = nn.Tanh
                else:
                    # fallback to ReLU for unknown strings
                    act_cls = nn.ReLU
            else:
                act_cls = mlp_activation
        
        # Create MLP if layer_dims or output_dim is specified
        layer_dims = [] if mlp_layer_dims is None else list(mlp_layer_dims)
        out_dim = in_dim if mlp_output_dim is None else int(mlp_output_dim)
        
        # Only create MLP if we're actually projecting to a different dimension
        # or if layer_dims is specified
        if mlp_output_dim is not None or (mlp_layer_dims is not None and len(mlp_layer_dims) > 0):
            self._mlp = BaseNets.MLP(
                input_dim=in_dim,
                output_dim=out_dim,
                layer_dims=layer_dims,
                activation=act_cls if act_cls is not None else nn.ReLU,
            )
            self._final_output_dim = out_dim
        else:
            self._mlp = None
            self._final_output_dim = in_dim

    def output_shape(self, input_shape):
        """
        Returns output shape for this module, which is a flat array if flatten=True, 
        otherwise it's the output of the visual backbone + pooling, potentially 
        projected through an MLP.
        """
        if self._mlp is not None:
            return [self._final_output_dim]
        else:
            return super(VisualCore_MSC, self).output_shape(input_shape)

    def forward(self, inputs):
        """
        Forward pass through visual core and optional MLP.
        """
        ndim = len(self.input_shape)
        assert tuple(inputs.shape)[-ndim:] == tuple(self.input_shape)
        
        # Pass through parent's visual core (backbone + pool + feature projection)
        x = self.nets(inputs)
        
        # Pass through MLP if it exists
        if self._mlp is not None:
            # Flatten if needed
            if len(x.shape) > 2:
                x = x.reshape(x.shape[0], -1)
            x = self._mlp(x)
        
        return x

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 2
        msg += textwrap.indent(
            "\ninput_shape={}\noutput_shape={}".format(self.input_shape, self.output_shape(self.input_shape)), indent)
        msg += textwrap.indent("\nbackbone_net={}".format(self.backbone), indent)
        msg += textwrap.indent("\npool_net={}".format(self.pool), indent)
        if self._mlp is not None:
            msg += textwrap.indent("\nmlp={}".format(self._mlp), indent)
        msg = header + '(' + msg + '\n)'
        return msg