import math

import torch
from torch import nn

from overcast.modules import dense


class ConditionalPreactivation(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_condition: int,
        dim_output: int,
        num_basis: int,
        dim_basis: int,
        bias: bool,
        negative_slope: float,
        dropout_rate: float,
        layer_norm: bool,
        spectral_norm: float,
    ) -> None:
        super(ConditionalPreactivation, self).__init__()
        self.activation = dense.DenseActivation(
            dim_input=dim_input,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            layer_norm=layer_norm,
        )
        linear = ConditionalLinear(
            dim_input=dim_input,
            dim_condition=dim_condition,
            dim_output=dim_output,
            num_basis=num_basis,
            basis_features=dim_basis,
            bias=bias,
        )
        self.linear = (
            dense.spectral_norm_fc(linear, spectral_norm)
            if spectral_norm > 0.0
            else linear
        )

    def forward(self, inputs: list[torch.Tensor]):
        x, c = inputs
        return self.linear([self.activation(x), c])


class ConditionalResidual(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_condition: int,
        dim_output: int,
        num_basis: int,
        dim_basis: int,
        bias: bool,
        negative_slope: float,
        dropout_rate: float,
        layer_norm: bool,
        spectral_norm: float,
    ) -> None:
        super(ConditionalResidual, self).__init__()
        if dim_input != dim_output:
            self.dropout = nn.Dropout(p=dropout_rate)
            linear = ConditionalLinear(
                dim_input=dim_input,
                dim_condition=dim_condition,
                dim_output=dim_output,
                num_basis=num_basis,
                basis_features=dim_basis,
                bias=bias,
            )
            self.shortcut = (
                dense.spectral_norm_fc(linear, spectral_norm)
                if spectral_norm > 0.0
                else linear
            )
        else:
            self.dropout = nn.Identity()
            self.shortcut = ConditionalIdentity()

        self.op = ConditionalPreactivation(
            dim_input=dim_input,
            dim_condition=dim_condition,
            dim_output=dim_output,
            num_basis=num_basis,
            dim_basis=dim_basis,
            bias=bias,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x, c = inputs
        return self.op(inputs) + self.shortcut([self.dropout(x), c])


MODULES = {"basic": ConditionalPreactivation, "resnet": ConditionalResidual}


class ConditionalFeatureExtractor(nn.Module):
    def __init__(
        self,
        architecture: str,
        dim_input: int,
        dim_condition: int,
        dim_hidden: int,
        depth: int,
        num_basis: int,
        dim_basis: int,
        negative_slope: float,
        dropout_rate: float,
        layer_norm: bool,
        spectral_norm: float,
        activate_output=True,
    ) -> None:
        super(ConditionalFeatureExtractor, self).__init__()
        self.ops = nn.ModuleList()
        hidden_module = MODULES[architecture]
        for i in range(depth):
            _dim_input = dim_input if i == 0 else dim_hidden
            self.ops.append(
                module=hidden_module(
                    dim_input=_dim_input,
                    dim_condition=dim_condition,
                    dim_output=dim_hidden,
                    num_basis=num_basis,
                    dim_basis=dim_basis,
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
        x, c = inputs
        for op in self.ops:
            x = op([x, c])
        return self.activation(x)


class ConditionalLinear(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_condition: int,
        dim_output: int,
        num_basis: int,
        basis_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(ConditionalLinear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_condition = dim_condition
        self.num_basis = num_basis
        self.basis_features = basis_features
        self.basis_functions = nn.Sequential(
            nn.Linear(in_features=dim_condition, out_features=basis_features),
            dense.DenseFeatureExtractor(
                architecture="resnet",
                dim_input=basis_features,
                dim_hidden=basis_features,
                depth=1,
                negative_slope=0.01,
                layer_norm=False,
                dropout_rate=0.0,
                spectral_norm=0.0,
                activate_output=True,
            ),
            nn.Linear(in_features=basis_features, out_features=num_basis),
        )
        self.weight = nn.parameter.Parameter(
            torch.empty((num_basis, dim_input, dim_output), **factory_kwargs)
        )
        if bias:
            self.bias = nn.parameter.Parameter(
                torch.empty((num_basis, dim_output), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.dim_input)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x, c = inputs  # (batch_size, dim_input), (batch_size, dim_condition)
        basis_functions = self.basis_functions(c)  # (batch_size, num_basis)
        w = torch.matmul(
            basis_functions,
            self.weight.view(self.num_basis, self.dim_input * self.dim_output),
        ).reshape(
            -1, self.dim_input, self.dim_output
        )  # (batch_size, dim_input, dim_output)
        output = torch.bmm(x.unsqueeze(1), w).squeeze(1)  # (batch_size, dim_output)
        if self.bias is not None:
            bias = torch.matmul(basis_functions, self.bias)  # (batch_size, dim_output)
            output += bias
        return output

    def extra_repr(self) -> str:
        return f"groups={self.num_groups}, dim_input={self.dim_input}, dim_output={self.dim_output}, bias={self.bias}"


class ConditionalIdentity(nn.Module):
    def __init__(self) -> None:
        super(ConditionalIdentity, self).__init__()

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x, c = inputs
        return x
