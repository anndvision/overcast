import math

import torch
from torch import nn

from overcast.modules import dense


class GroupPreactivation(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_groups: int,
        bias: bool,
        negative_slope: float,
        dropout_rate: float,
        layer_norm: bool,
        spectral_norm: float,
    ) -> None:
        super(GroupPreactivation, self).__init__()
        self.activation = dense.DenseActivation(
            dim_input=dim_input,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            layer_norm=layer_norm,
        )
        linear = GroupLinear(
            dim_input=dim_input,
            dim_output=dim_output,
            num_groups=num_groups,
            bias=bias,
        )
        self.linear = (
            dense.spectral_norm_fc(linear, spectral_norm)
            if spectral_norm > 0.0
            else linear
        )

    def forward(self, inputs: list[torch.Tensor]):
        x, g = inputs
        return self.linear([self.activation(x), g])


class GroupResidual(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_groups: int,
        bias: bool,
        negative_slope: float,
        dropout_rate: float,
        layer_norm: bool,
        spectral_norm: float,
    ) -> None:
        super(GroupResidual, self).__init__()
        if dim_input != dim_output:
            self.dropout = nn.Dropout(p=dropout_rate)
            linear = GroupLinear(
                dim_input=dim_input,
                dim_output=dim_output,
                num_groups=num_groups,
                bias=bias,
            )
            self.shortcut = (
                dense.spectral_norm_fc(linear, spectral_norm)
                if spectral_norm > 0.0
                else linear
            )
        else:
            self.dropout = nn.Identity()
            self.shortcut = GroupIdentity()

        self.op = GroupPreactivation(
            dim_input=dim_input,
            dim_output=dim_output,
            num_groups=num_groups,
            bias=bias,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x, g = inputs
        return self.op(inputs) + self.shortcut([self.dropout(x), g])


MODULES = {"basic": GroupPreactivation, "resnet": GroupResidual}


class GroupFeatureExtractor(nn.Module):
    def __init__(
        self,
        architecture: str,
        dim_input: int,
        dim_hidden: int,
        groups: int,
        depth: int,
        negative_slope: float,
        dropout_rate: float,
        layer_norm: bool,
        spectral_norm: float,
        activate_output=True,
    ) -> None:
        super(GroupFeatureExtractor, self).__init__()
        self.ops = nn.ModuleList()
        hidden_module = MODULES[architecture]
        for i in range(depth):
            _dim_input = dim_input if i == 0 else dim_hidden
            self.ops.append(
                module=hidden_module(
                    dim_input=_dim_input,
                    dim_output=dim_hidden,
                    groups=groups,
                    bias=not layer_norm,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    layer_norm=layer_norm,
                    spectral_norm=spectral_norm,
                )
            )
        if activate_output:
            self.activation = dense.DenseActivation(
                dim_input=dim_hidden,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
            )
        else:
            self.activation = nn.Identity()
        self.dim_output = dim_hidden

    def forward(self, inputs: list[torch.Tensor]):
        x, g = inputs
        for op in self.ops:
            x = op([x, g])
        return self.activation(x)


class GroupLinear(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_output: int,
        num_groups: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(GroupLinear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_groups = num_groups
        self.weight = nn.parameter.Parameter(
            torch.empty((self.num_groups, dim_input, dim_output), **factory_kwargs)
        )
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.empty((self.num_groups, dim_output), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.dim_input)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x, g = inputs
        w = torch.matmul(
            g, self.weight.view(self.num_groups, self.dim_input * self.dim_output),
        ).reshape(-1, self.dim_input, self.dim_output)
        return torch.bmm(x.unsqueeze(1), w).squeeze(1) + torch.matmul(g, self.bias)

    def extra_repr(self) -> str:
        return f"num_groups={self.num_groups}, dim_input={self.dim_input}, dim_output={self.dim_output}, bias={self.bias}"


class GroupIdentity(nn.Module):
    def __init__(self) -> None:
        super(GroupIdentity, self).__init__()

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x, g = inputs
        return x
