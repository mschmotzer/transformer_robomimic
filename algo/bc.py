"""
Implementation of Behavioral Cloning (BC).
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


@register_algo_factory_func("bc")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    # note: we need the check below because some configs import BCConfig and exclude
    # some of these options
    gaussian_enabled = ("gaussian" in algo_config and algo_config.gaussian.enabled)
    gmm_enabled = ("gmm" in algo_config and algo_config.gmm.enabled)
    vae_enabled = ("vae" in algo_config and algo_config.vae.enabled)
    # Get action_chunk_size from transformer config if available
    action_chunk_size = 1  # default value
    if hasattr(algo_config, 'transformer') and hasattr(algo_config.transformer, 'action_chunk_size'):
        action_chunk_size = algo_config.transformer.action_chunk_size
    elif hasattr(algo_config, 'action_chunk_size'):
        action_chunk_size = algo_config.action_chunk_size

    rnn_enabled = algo_config.rnn.enabled
    transformer_enabled = algo_config.transformer.enabled

    if gaussian_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_Gaussian, {}
    elif gmm_enabled:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN_GMM, {}
        elif transformer_enabled:
            if action_chunk_size == 1:
                algo_class, algo_kwargs = BC_Transformer_GMM, {}
                print("Using BC_Transformer_GMM")
            else:
                algo_class, algo_kwargs = BC_Transformer_GMM_CHUNKING, {}
                print("Using BC_Transformer_GMM_CHUNKING")

        else:
            algo_class, algo_kwargs = BC_GMM, {}
    elif vae_enabled:
        if rnn_enabled:
            raise NotImplementedError
        elif transformer_enabled:
            raise NotImplementedError
        else:
            algo_class, algo_kwargs = BC_VAE, {}
    else:
        if rnn_enabled:
            algo_class, algo_kwargs = BC_RNN, {}
        elif transformer_enabled:
            if action_chunk_size == 1:
                algo_class, algo_kwargs = BC_Transformer, {}
                print("Using BC_Transformer")
            else:
                algo_class, algo_kwargs = BC_Transformer_CHUNKING, {}
                print("Using BC_Transformer_CHUNKING")
        else:
            algo_class, algo_kwargs = BC, {}

    return algo_class, algo_kwargs


class BC(PolicyAlgo):
    """
    Normal BC training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.ActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))


    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(BC, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)

        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        predictions = OrderedDict()
        actions = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"] = actions
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss
        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(BC, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item()
        if "l2_loss" in info["losses"]:
            log["L2_Loss"] = info["losses"]["l2_loss"].item()
        if "l1_loss" in info["losses"]:
            log["L1_Loss"] = info["losses"]["l1_loss"].item()
        if "cos_loss" in info["losses"]:
            log["Cosine_Loss"] = info["losses"]["cos_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        return self.nets["policy"](obs_dict, goal_dict=goal_dict)


class BC_Gaussian(BC):
    """
    BC training with a Gaussian policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gaussian.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GaussianActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            fixed_std=self.algo_config.gaussian.fixed_std,
            init_std=self.algo_config.gaussian.init_std,
            std_limits=(self.algo_config.gaussian.min_std, 7.5),
            std_activation=self.algo_config.gaussian.std_activation,
            low_noise_eval=self.algo_config.gaussian.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_GMM(BC_Gaussian):
    """
    BC training with a Gaussian Mixture Model policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)


class BC_VAE(BC):
    """
    BC training with a VAE policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.VAEActor(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            device=self.device,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **VAENets.vae_args_from_config(self.algo_config.vae),
        )
        
        self.nets = self.nets.float().to(self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Update from superclass to set categorical temperature, for categorical VAEs.
        """
        if self.algo_config.vae.prior.use_categorical:
            temperature = self.algo_config.vae.prior.categorical_init_temp - epoch * self.algo_config.vae.prior.categorical_temp_anneal_step
            temperature = max(temperature, self.algo_config.vae.prior.categorical_min_temp)
            self.nets["policy"].set_gumbel_temperature(temperature)
        return super(BC_VAE, self).train_on_batch(batch, epoch, validate=validate)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        vae_inputs = dict(
            actions=batch["actions"],
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
            freeze_encoder=batch.get("freeze_encoder", False),
        )

        vae_outputs = self.nets["policy"].forward_train(**vae_inputs)
        predictions = OrderedDict(
            actions=vae_outputs["decoder_outputs"],
            kl_loss=vae_outputs["kl_loss"],
            reconstruction_loss=vae_outputs["reconstruction_loss"],
            encoder_z=vae_outputs["encoder_z"],
        )
        if not self.algo_config.vae.prior.use_categorical:
            with torch.no_grad():
                encoder_variance = torch.exp(vae_outputs["encoder_params"]["logvar"])
            predictions["encoder_variance"] = encoder_variance
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # total loss is sum of reconstruction and KL, weighted by beta
        kl_loss = predictions["kl_loss"]
        recons_loss = predictions["reconstruction_loss"]
        action_loss = recons_loss + self.algo_config.vae.kl_weight * kl_loss
        return OrderedDict(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["KL_Loss"] = info["losses"]["kl_loss"].item()
        log["Reconstruction_Loss"] = info["losses"]["recons_loss"].item()
        if self.algo_config.vae.prior.use_categorical:
            log["Gumbel_Temperature"] = self.nets["policy"].get_gumbel_temperature()
        else:
            log["Encoder_Variance"] = info["predictions"]["encoder_variance"].mean().item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN(BC):
    """
    BC training with an RNN policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = batch["obs"]
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"]

        if self._rnn_is_open_loop:
            # replace the observation sequence with one that only consists of the first observation.
            # This way, all actions are predicted "open-loop" after the first observation, based
            # on the rnn hidden state.
            n_steps = batch["actions"].shape[1]
            obs_seq_start = TensorUtils.index_at_time(batch["obs"], ind=0)
            input_batch["obs"] = TensorUtils.unsqueeze_expand_at(obs_seq_start, size=n_steps, dim=1)

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        """print("Obs To USE--------:", obs_to_use)
        print("RNN HIDDEN STATE--------:", len(self._rnn_hidden_state))
        print("Goal Dict--------:", goal_dict)"""
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0


class BC_RNN_GMM(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.RNNGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"],
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2 # [B, T]
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_Transformer(BC):
    """
    BC training with a Transformer policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)
        
    def _set_params_from_config(self):
        """
        Read specific config variables we need for training / eval.
        Called by @_create_networks method
        """
        self.context_length = self.algo_config.transformer.context_length
        self.supervise_all_steps = self.algo_config.transformer.supervise_all_steps

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            input_batch["actions"] = batch["actions"][:, :h, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h-1, :]

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch

    def _forward_training(self, batch, epoch=None):
        """
        Internal helper function for BC_Transformer algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )

        predictions = OrderedDict()
        predictions["actions"] = self.nets["policy"](obs_dict=batch["obs"], actions=None, goal_dict=batch["goal_obs"])
        if not self.supervise_all_steps:
            # only supervise final timestep
            predictions["actions"] = predictions["actions"][:, -1, :]
        return predictions

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.
        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal
        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        print("obs:", obs_dict)
        obs_seq = {}
        for key, value in obs_dict.items():
            if value.ndim == 2:  # [B, features]
                obs_seq[key] = value.unsqueeze(1)  # [B, 1, features]
            else:
                obs_seq[key] = value
        
        actions = self.nets["policy"](obs_seq, actions=None, goal_dict=goal_dict)[:, -1, :]  
        return actions

class BC_Transformer_CHUNKING(BC_Transformer):
    """
    BC training with a Transformer policy predicting chunks of actions.
    Includes temporal action buffer ensemble for smooth control.
    """

    def _create_networks(self):
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerActorChunkingNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            action_chunk_size=self.algo_config.transformer.action_chunk_size,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Obs → actions shifted into the future by chunk_size
        """
        input_batch = {}
        h = self.context_length
        c = self.algo_config.transformer.action_chunk_size

        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None)

        # Ground truth future chunk
        input_batch["actions"] = batch["actions"][:, h:h+c, :]  # [B, C, ac_dim]
        # Ground truth future chunk or supervised full sequence (context + chunk)
        if self.supervise_all_steps:
            # supervise on entire sequence: first h timesteps (context) followed by future chunk
            input_batch["actions"] = batch["actions"][:, :h + c, :]  # [B, h+C, ac_dim]
        else:
            # only ground-truth future chunk
            input_batch["actions"] = batch["actions"][:, h:h + c, :]  # [B, C, ac_dim]

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)


    def _forward_training(self, batch, epoch=None):
        TensorUtils.assert_size_at_dim(batch["obs"], size=self.context_length, dim=1,msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length))
            
        pred = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"], supervised=self.supervise_all_steps,
        )
        return OrderedDict(actions=pred)

   # def _compute_losses(self, predictions, batch):
   #     print("PREDICTIONS ACTIONS SHAPE:", predictions["actions"][0,0,:])
   #     print("BATCH ACTIONS SHAPE:", batch["actions"][0,0,:])
   #     return OrderedDict(action_loss=nn.MSELoss()(
   #         predictions["actions"], batch["actions"]
   #     ))

    def _compute_losses(self, predictions, batch):
        pred = predictions["actions"]
        gt   = batch["actions"]

        # ---- split actions ----
        pos_pred = pred[..., :3]
        pos_gt   = gt[..., :3]

        quat_pred = pred[..., 3:7]
        quat_gt   = gt[..., 3:7]

        grip_pred = pred[..., 7]
        grip_gt   = gt[..., 7]

        # ---- position loss (metric-aligned, meters) ----
        # Mean Euclidean distance
        pos_loss = torch.norm(pos_pred - pos_gt, dim=-1).mean()

        # ---- quaternion loss (geodesic) ----
        quat_pred = F.normalize(quat_pred, dim=-1)
        quat_gt   = F.normalize(quat_gt, dim=-1)
        dot = torch.abs(torch.sum(quat_pred * quat_gt, dim=-1))
        quat_loss = (1.0 - dot).mean()

        # ---- gripper loss ----
        # assumes continuous or {-1,1}
        grip_loss = F.mse_loss(grip_pred, grip_gt)

        # ---- total loss (weights matter!) ----
        action_loss = (
            1.0 * pos_loss +
            0.1 * quat_loss +
            0.01 * grip_loss
        ) 
        with torch.no_grad():
            pos_err_cm = pos_loss * 100.0

        return OrderedDict(
            action_loss=action_loss,
            pos_loss=pos_loss,
            quat_loss=quat_loss,
            grip_loss=grip_loss,
            pos_err_cm=pos_err_cm,
        )

    def reset(self):
        """
        Clear temporal action buffer.
        """
        self._action_buffer = []

    def get_action(self, obs_dict, goal_dict=None):
        """
        Weighted temporal ensemble across chunk buffer.
        """
        assert not self.nets.training
        
        # Init temporal buffer if needed
        if not hasattr(self, "_action_buffer"):
            self.reset()

        # Expand obs to include time dim
        obs_seq = {k: v.unsqueeze(1) if v.ndim == 2 else v for k, v in obs_dict.items()}

        # Predict a new chunk: [1, C, ac_dim]
        chunk = self.nets["policy"](obs_seq, actions=None, goal_dict=goal_dict).squeeze(0)
        C = chunk.shape[0]

        # Add newest chunk to buffer
        self._action_buffer.append(chunk)
        max_buf = self.algo_config.transformer.action_chunk_size
        if len(self._action_buffer) > max_buf:
            self._action_buffer.pop(0)

        # Ensemble current action
        actions = []
        weights = []
        N = len(self._action_buffer)

        for i, ch in enumerate(self._action_buffer):
            # Temporal alignment:
            # Newest buffer entry predicts time ≈ 0
            idx = N - 1 - i  # older entries weighted higher
            if 0 <= idx < C:
                actions.append(ch[idx])
                weights.append(N - i)  # older = stronger belief

        if len(actions) == 0:
            return torch.zeros(1, self.ac_dim, device=chunk.device)

        actions = torch.stack(actions)  # [num_valid, ac_dim]
        weights = torch.tensor(weights, device=actions.device, dtype=torch.float)
        weights = weights / weights.sum()

        action = (actions * weights.unsqueeze(1)).sum(dim=0)  # weighted mean
        return action.unsqueeze(0)  # [1, ac_dim]

class BC_Transformer_GMM(BC_Transformer):
    """
    BC training with a Transformer GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerGMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),
        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch, epoch=None):
        """
        Modify from super class to support GMM training.
        """
        # ensure that transformer context length is consistent with temporal dimension of observations
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            actions=None,
            goal_dict=batch["goal_obs"],
            low_noise_eval=False,
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        #print("DISTS BATCH SHAPE:", dists.batch_shape)
        assert len(dists.batch_shape) == 2 # [B, T]

        if not self.supervise_all_steps:
            # only use final timestep prediction by making a new distribution with only final timestep.
            # This essentially does `dists = dists[:, -1]`
            component_distribution = D.Normal(
                loc=dists.component_distribution.base_dist.loc[:, -1],
                scale=dists.component_distribution.base_dist.scale[:, -1],
            )
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=dists.mixture_distribution.logits[:, -1])
            dists = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC_Transformer_GMM algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.
        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training
        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = -predictions["log_probs"].mean()
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.
        Args:
            info (dict): dictionary of info
        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item() 
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
class BC_Transformer_GMM_CHUNKING(BC_Transformer_GMM):
    """
    BC training with a Transformer GMM policy.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.transformer.enabled
        print("-----------------Training to create Chunking-----------------")
        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.TransformerGMMActorNetworkActionChunking(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            num_modes=self.algo_config.gmm.num_modes,
            action_chunk_size=self.algo_config.transformer.action_chunk_size,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.transformer_args_from_config(self.algo_config.transformer),

        )
        self._set_params_from_config()
        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.
        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader
        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        chunk_size = self.algo_config.transformer.action_chunk_size
        h = self.context_length
        input_batch["obs"] = {k: batch["obs"][k][:, :h, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present

        if self.supervise_all_steps:
            # supervision on entire sequence (instead of just current timestep)
            input_batch["actions"] = batch["actions"][:, :h+chunk_size, :]
        else:
            # just use current timestep
            input_batch["actions"] = batch["actions"][:, h:h+chunk_size, :]

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
        return input_batch
    
    """def process_batch_for_training(self, batch):
        
        Create obs-action pairs for action chunking time series prediction.
        Maps obs[T-context:T] -> actions[T:T+chunk_size]
        
        input_batch = dict()
        h = self.context_length  # e.g., 40 (observation window size)
        chunk_size = self.algo_config.transformer.action_chunk_size  # e.g., 32 (future predictions)
        
        B, T, ac_dim = batch["actions"].shape
        
        # Need enough sequence length for context + future chunk
        if T < h + chunk_size:
            raise ValueError(f"Sequence length {T} too short for context {h} + chunk {chunk_size}")
        
        # Create all valid sliding windows
        # We want windows where we can predict chunk_size steps into the future
        max_start_time = T - h - chunk_size  # Last valid starting position
        
        all_obs = []
        all_actions = []
        
        # Create training examples from sliding windows
        for t in range(max_start_time + 1):  # t is the prediction time
            # Observation window: [t-h:t] (context before prediction time)
            obs_start = max(0, t)
            obs_end = min(t + h, T)
            
            # Pad if necessary at the beginning
            if obs_start < h:
                obs_window = {}
                for k in batch["obs"].keys():
                    obs_data = batch["obs"][k][:, obs_start:obs_end, :]  # [B, actual_len, dim]
                    
                    # Pad at the beginning if needed
                    actual_len = obs_end - obs_start
                    if actual_len < h:
                        pad_len = h - actual_len
                        pad_shape = list(obs_data.shape)
                        pad_shape[1] = pad_len
                        padding = torch.zeros(pad_shape, dtype=obs_data.dtype, device=obs_data.device)
                        obs_data = torch.cat([padding, obs_data], dim=1)
                    
                    obs_window[k] = obs_data
            else:
                # Normal case: full context window
                obs_window = {k: batch["obs"][k][:, t:t+h, :] for k in batch["obs"]}
            
            # Future actions: [t+h:t+h+chunk_size]
            action_start = t + h
            action_end = action_start + chunk_size
            
            if action_end <= T:
                # Full chunk available
                action_chunk = batch["actions"][:, action_start:action_end, :]  # [B, chunk_size, ac_dim]
            else:
                # Partial chunk - pad with last available action
                available_actions = batch["actions"][:, action_start:T, :]
                available_len = T - action_start
                
                if available_len > 0:
                    # Pad with last action
                    last_action = batch["actions"][:, -1:, :].repeat(1, chunk_size - available_len, 1)
                    action_chunk = torch.cat([available_actions, last_action], dim=1)
                else:
                    # No future actions available - skip this window
                    continue
            
            all_obs.append(obs_window)
            all_actions.append(action_chunk.view(B, chunk_size * ac_dim))  # Flatten for GMM
        
        if len(all_obs) == 0:
            raise ValueError("No valid training windows could be created")
        
        # Concatenate all windows into batch
        input_batch["obs"] = {}
        for k in batch["obs"].keys():
            input_batch["obs"][k] = torch.cat([obs[k] for obs in all_obs], dim=0)
        
        input_batch["actions"] = torch.cat(all_actions, dim=0)
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        
        # Expand goals to match new batch size if present
        if input_batch["goal_obs"] is not None:
            num_windows = len(all_obs)
            input_batch["goal_obs"] = {
                k: v.repeat(num_windows, 1, 1) for k, v in input_batch["goal_obs"].items()
            }

        input_batch = TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        return input_batch"""
    

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        TensorUtils.assert_size_at_dim(
            batch["obs"], 
            size=(self.context_length), 
            dim=1, 
            msg="Error: expect temporal dimension of obs batch to match transformer context length {}".format(self.context_length),
        )
        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"], 
            goal_dict=batch["goal_obs"], supervised=self.supervise_all_steps,
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        """predictions=[]
        for dist in dists:
            print("DIST SHAPE: ", dist.batch_shape)
            assert len(dist.batch_shape) == 2 # [B, T]
            log_probs = dist.log_prob(batch["actions"])

            prediction = OrderedDict(
                log_probs=log_probs,
            )
            predictions.append(prediction)"""
        assert len(dists.batch_shape) == 2 # [B, T]
        #print("DISTS BATCH SHAPE:", dists.batch_shape)
        #print("BATCH ACTIONS SHAPE: ", batch["actions"].shape)
        #print("ACTION CHUNK'S SIZE: ", self.algo_config.transformer.action_chunk_size)
        log_probs = dists.log_prob(batch["actions"])
        prediction = OrderedDict(
            log_probs=log_probs,
        )
      
        return prediction

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        action_loss = 0
        #for pred in predictions:
        action_loss -= predictions["log_probs"].mean()   
        return OrderedDict(
            log_probs=-action_loss,
            action_loss=action_loss,
        )


    def get_action(self, obs_dict, goal_dict=None):
        """
        Get action using weighted temporal ensemble with action buffer.
        Buffer fills with action chunks, actions are weighted by age (older = higher weight).
        """
        assert not self.nets.training
        
        # Initialize action buffer if needed
        if not hasattr(self, '_action_buffer'):
            self._action_buffer = []  # List of action chunks [chunk_size, ac_dim]
            self._buffer_size = self.algo_config.transformer.action_chunk_size
        
        # Always predict new action chunk
        # Add sequence dimension for transformer input
        obs_seq = {}
        for key, value in obs_dict.items():
            if value.ndim == 2:  # [B, features]
                obs_seq[key] = value.unsqueeze(1)  # [B, 1, features]
            else:
                obs_seq[key] = value
        
        # Get new action chunk prediction: [B, T, chunk_size, ac_dim]

        action_chunks = self.nets["policy"](obs_seq, actions=None, goal_dict=goal_dict)
        #print("ACTION CHUNKS: ", action_chunks.shape)
        new_chunk = action_chunks[:, :, :].squeeze(0)  # [chunk_size, ac_dim] - take last timestep
        
        # Add new chunk to buffer
        self._action_buffer.append(new_chunk)
        # Keep buffer size limited
        if len(self._action_buffer) > self._buffer_size:
            self._action_buffer.pop(0)  # Remove oldest
        
        # Compute weighted ensemble for current timestep
        if len(self._action_buffer) == 0:
            # Fallback
            return torch.zeros(1, self.ac_dim, device=obs_dict[list(obs_dict.keys())[0]].device)
        
        # Collect actions for current timestep from all buffer entries
        current_timestep_actions = []
        weights = []
        
        for buffer_idx, chunk in enumerate(self._action_buffer):
            # Each buffer entry is shifted by its position
            # buffer_idx=0 (oldest) predicts timestep 0, 1, 2, ...
            # buffer_idx=1 predicts timestep 0, 1, 2, ... (but offset by 1)
            # For current timestep (always 0), we want:
            # - From buffer_idx=0: action at index (len(buffer)-1-buffer_idx)
            # - From buffer_idx=1: action at index (len(buffer)-1-buffer_idx)
        
            action_idx = len(self._action_buffer) - 1 - buffer_idx
            
            if 0 <= action_idx < chunk.shape[0]:
                action = chunk[action_idx, :]  # [ac_dim]
                current_timestep_actions.append(action)
                
                # Weight: older predictions (smaller buffer_idx) get higher weight
                weight = len(self._action_buffer) - buffer_idx
                weights.append(weight)
        
        if len(current_timestep_actions) == 0:
            # Fallback
            return torch.zeros(1, self.ac_dim, device=obs_dict[list(obs_dict.keys())[0]].device)
        
        # Compute weighted average
        actions_tensor = torch.stack(current_timestep_actions, dim=0)  # [num_predictions, ac_dim]
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=actions_tensor.device)
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize weights
        
        # Weighted average
        weighted_action = (actions_tensor * weights_tensor.unsqueeze(1)).sum(dim=0)  # [ac_dim]
        #print("CHUNKS: ", action_chunks.shape)
        #print("Weighted Action: ", weighted_action)
        return weighted_action.unsqueeze(0)  # [1, ac_dim]

    def reset(self):
        """
        Reset action buffer.
        """
        if hasattr(self, '_action_buffer'):
            self._action_buffer = []
            

