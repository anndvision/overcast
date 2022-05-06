from torch import nn

from overcast.modules.spectral_norm import spectral_norm_fc


class DenseActivation(nn.Module):
    def __init__(
        self, dim_input, negative_slope, dropout_rate, layer_norm,
    ):
        super(DenseActivation, self).__init__()
        self.op = nn.Sequential(
            nn.LayerNorm(dim_input) if layer_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=negative_slope)
            if negative_slope >= 0.0
            else nn.GELU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, inputs):
        return self.op(inputs)


class DensePreactivation(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        bias,
        negative_slope,
        dropout_rate,
        layer_norm,
        spectral_norm,
    ):
        super(DensePreactivation, self).__init__()
        self.op = nn.Sequential(
            DenseActivation(
                dim_input=dim_input,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
            )
        )
        linear = nn.Linear(in_features=dim_input, out_features=dim_output, bias=bias)
        self.op.add_module(
            "linear",
            spectral_norm_fc(linear, spectral_norm) if spectral_norm > 0.0 else linear,
        )

    def forward(self, inputs):
        return self.op(inputs)


class DenseResidual(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        bias,
        negative_slope,
        dropout_rate,
        layer_norm,
        spectral_norm,
    ):
        super(DenseResidual, self).__init__()
        if dim_input != dim_output:
            self.shortcut = nn.Sequential(nn.Dropout(p=dropout_rate))
            linear = nn.Linear(
                in_features=dim_input, out_features=dim_output, bias=bias
            )
            self.shortcut.add_module(
                "linear",
                spectral_norm_fc(linear, spectral_norm)
                if spectral_norm > 0.0
                else linear,
            )
        else:
            self.shortcut = nn.Identity()

        self.op = DensePreactivation(
            dim_input=dim_input,
            dim_output=dim_output,
            bias=bias,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
        )

    def forward(self, inputs):
        return self.op(inputs) + self.shortcut(inputs)


MODULES = {"basic": DensePreactivation, "resnet": DenseResidual}


class DenseLinear(nn.Module):
    def __init__(
        self, dim_input, dim_output, layer_norm, spectral_norm,
    ):
        super(DenseLinear, self).__init__()
        self.op = nn.Linear(
            in_features=dim_input, out_features=dim_output, bias=not layer_norm,
        )
        if spectral_norm > 0.0:
            self.op = spectral_norm_fc(self.op, spectral_norm)

    def forward(self, inputs):
        return self.op(inputs)


class DenseFeatureExtractor(nn.Module):
    def __init__(
        self,
        architecture,
        dim_input,
        dim_hidden,
        depth,
        negative_slope,
        layer_norm,
        dropout_rate,
        spectral_norm,
        activate_output=True,
    ):
        super(DenseFeatureExtractor, self).__init__()
        self.op = nn.Sequential()
        hidden_module = MODULES[architecture]
        if depth == 0:
            self.op.add_module(
                name="hidden_layer_0", module=nn.Identity(),
            )
        else:
            for i in range(depth):
                _dim_input = dim_input if i == 0 else dim_hidden
                self.op.add_module(
                    name="hidden_layer_{}".format(i),
                    module=hidden_module(
                        dim_input=_dim_input,
                        dim_output=dim_hidden,
                        bias=not layer_norm,
                        negative_slope=negative_slope,
                        dropout_rate=dropout_rate,
                        layer_norm=layer_norm,
                        spectral_norm=spectral_norm,
                    ),
                )
        if activate_output:
            self.op.add_module(
                name="output_activation",
                module=DenseActivation(
                    dim_input=dim_hidden,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    layer_norm=layer_norm,
                ),
            )
        self.dim_output = dim_hidden

    def forward(self, inputs):
        return self.op(inputs)
