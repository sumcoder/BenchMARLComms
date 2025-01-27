#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Type

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP, MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.discrete_vit import Transformer

import threading

class Mlp(Model):
    """Multi layer perceptron model.

    Args:
        num_cells (int or Sequence[int], optional): number of cells of every layer in between the input and output. If
            an integer is provided, every layer will have the same number of cells. If an iterable is provided,
            the linear layers out_features will match the content of num_cells.
        layer_class (Type[nn.Module]): class to be used for the linear layers;
        activation_class (Type[nn.Module]): activation class to be used.
        activation_kwargs (dict, optional): kwargs to be used with the activation class;
        norm_class (Type, optional): normalization class, if any.
        norm_kwargs (dict, optional): kwargs to be used with the normalization layers;

    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
        )

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        dim = 2 * (self.n_agents - 1) + 4 + 12  # 4 is for agent own pos and vel, 12 is sensor const
        self.transformer = Transformer(0.001, dim=dim, depth=1, heads=6, qk_dim=64, v_dim=64,
                                       mlp_dim=1536, dropout=0)

        if self.input_has_agent_dim:
            print("input has agent dim") 
            self.mlp = MultiAgentMLP(
                n_agent_inputs=self.input_features,
                n_agent_outputs=self.output_features,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **kwargs,
            )
        else:
            print("input has no agent dim")
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_features=self.input_features,
                        out_features=self.output_features,
                        device=self.device,
                        **kwargs,
                    )
                    for _ in range(self.n_agents if not self.share_params else 1)
                ]
            )


    def _perform_checks(self):
        super()._perform_checks()

        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError(
                "If the MLP input has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )


    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = tensordict.get(self.in_key)

        # Has multi-agent input dimension
        if self.input_has_agent_dim:
            # adding a graph neural network to enable communications
            aggregated_messages, info = (self.transformer(input))
            res = self.mlp.forward(aggregated_messages)
            # res = self.mlp.forward(input)
            if not self.output_has_agent_dim:
                # If we are here the module is centralised and parameter shared.
                # Thus the multi-agent dimension has been expanded,
                # We remove it without loss of data
                res = res[..., 0, :]

        # Does not have multi-agent input dimension
        else:
            if not self.share_params:
                res = torch.stack(
                    [net(input) for net in self.mlp],
                    dim=-2,
                )
            else:
                res = self.mlp[0](input)

        num_bits = torch.zeros_like(res)
        # print(res.shape)
        if info:
            num_bits[...,0,0] = info['k_reg_loss']
            if num_bits.shape[-1] == 1:
                num_bits[...,0,1,0] = info['wv_reg_loss']
            else:
                num_bits[..., 0, 1] = info['wv_reg_loss']
        tensordict.set(self.out_key, res)  # TODO: add num bits here and debug backwards
        tensordict.set('bits', num_bits)
        return tensordict


@dataclass
class MlpConfig(ModelConfig):
    """Dataclass config for a :class:`~benchmarl.models.Mlp`."""

    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    @staticmethod
    def associated_class():
        return Mlp
