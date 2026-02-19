import numpy as np
import torch.nn as nn
import torch
from copy import deepcopy
from gym import spaces
import rl_games.common.divergence as divergence
from rl_games.common.extensions.distributions import CategoricalMasked
from lib.utils.torch_utils import recurse_freeze, freeze_batchnorm_stats
from lib.rl.moving_avg import RunningMeanStd, RunningMeanStdObs
from .models import BaseModel, BaseModelNetwork
from .network_builder_transformer_bih_graph_improve_correct import BimanualHandToolPolicy


class ModelA2CContinuousLogStdBiH(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, "a2c")
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
            #self.transformer_policy = BimanualHandToolPolicy()

        def train(self, mode: bool = True):
            self.training = mode
            for name, module in self.named_children():
                if name in ["base_model", "lh_base_model", "rh_base_model"]:
                    module.train(False)  # always eval
                else:
                    module.train(mode)
            return self

        def load_state_dict(self, state_dict):
            return super().load_state_dict(state_dict)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            #mu, logstd, value, states, base_action = self.transformer_policy(input_dict['obs'])
            mu, logstd, value, states, base_action = self.a2c_network(input_dict['obs'])
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                    "base_actions": base_action,
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                + logstd.sum(dim=-1)
            )
