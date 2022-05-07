import math
import torch

from torch import nn

from overcast.modules import dense
from overcast.modules.spectral_norm import spectral_norm_fc


class Attention(nn.Module):
    def __init__(self, dim_input: int):
        super(Attention, self).__init__()
        self.dim_input = dim_input

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        scores = q.bmm(k.transpose(1, 2)) / math.sqrt(self.dim_input)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 0.0)
        att = torch.softmax(scores, -1) if mask.sum() > 0 else torch.zeros_like(scores)
        return att.bmm(v)


class ResAttention(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int,
        layer_norm: bool,
        dropout_rate: float,
        spectral_norm: float = 0.0,
    ):
        super(ResAttention, self).__init__()
        self.attention = (
            MultiHeadAttention(
                dim_q=dim_q, dim_k=dim_k, dim_v=dim_v, num_heads=num_heads,
            )
            if num_heads > 1
            else Attention(dim_q)
        )
        self.drop = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(dim_v) if layer_norm else nn.Identity()
        self.shortcut = nn.Linear(dim_q, dim_v) if dim_q != dim_v else nn.Identity()
        if spectral_norm > 0 and dim_q != dim_v:
            self.shortcut = spectral_norm_fc(self.shortcut, coeff=spectral_norm)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor = None,
    ) -> torch.Tensor:
        mha = self.attention(q, k, k, mask)
        return self.layer_norm(self.shortcut(q) + self.drop(mha))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int,
        spectral_norm: float = 0.0,
    ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.v_channels = dim_v
        self.fc_q = nn.Linear(dim_q, dim_v, bias=False)
        self.fc_k = nn.Linear(dim_k, dim_v, bias=False)
        self.fc_v = nn.Linear(dim_k, dim_v, bias=False)
        self.attention = Attention(dim_input=dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v, bias=False)
        if spectral_norm > 0:
            self.fc_q = spectral_norm_fc(self.fc_q, coeff=spectral_norm)
            self.fc_k = spectral_norm_fc(self.fc_k, coeff=spectral_norm)
            self.fc_v = spectral_norm_fc(self.fc_v, coeff=spectral_norm)
            self.fc_o = spectral_norm_fc(self.fc_o, coeff=spectral_norm)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.repeat_interleave(self.num_heads, dim=0)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        split_channels = self.v_channels // self.num_heads

        mha = self.attention(
            q=torch.cat(q.split(split_channels, -1), 0),
            k=torch.cat(k.split(split_channels, -1), 0),
            v=torch.cat(v.split(split_channels, -1), 0),
            mask=mask,
        )
        mha = torch.cat(mha.split(q.shape[0], 0), -1)
        return self.fc_o(mha)


class MAB(nn.Module):
    def __init__(
        self,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int,
        layer_norm: bool = False,
        negative_slope: float = 0.0,
        dropout_rate: float = 0.0,
        spectral_norm: float = 0.0,
    ):
        super(MAB, self).__init__()
        self.attention = ResAttention(
            dim_q=dim_q,
            dim_k=dim_k,
            dim_v=dim_v,
            num_heads=num_heads,
            layer_norm=layer_norm,
            dropout_rate=dropout_rate,
        )
        lin_1 = nn.Linear(dim_v, 4 * dim_v, bias=True)
        lin_2 = nn.Linear(4 * dim_v, dim_v, bias=True)
        self.fc_o = nn.Sequential(
            spectral_norm_fc(lin_1, coeff=spectral_norm)
            if spectral_norm > 0
            else lin_1,
            nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
            if negative_slope >= 0.0
            else nn.GELU(),
            spectral_norm_fc(lin_2, coeff=spectral_norm)
            if spectral_norm > 0
            else lin_2,
            nn.Dropout(p=dropout_rate),
        )
        self.norm = nn.LayerNorm(dim_v) if layer_norm else nn.Identity()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        output = self.attention(q, k, mask)
        output = output + self.fc_o(output)
        output = self.norm(output)
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        depth: int,
        num_heads: int,
        layer_norm: bool = True,
        negative_slope: float = 0.0,
        dropout_rate: float = 0.0,
        spectral_norm: float = 0.0,
    ):
        super(Encoder, self).__init__()

        self.input_embedding = nn.Sequential(
            dense.DenseLinear(
                dim_input=dim_input,
                dim_output=dim_hidden,
                layer_norm=False,
                spectral_norm=spectral_norm,
            ),
            nn.Dropout(p=dropout_rate),
        )
        self.position_embedding = nn.Sequential(
            dense.DenseLinear(
                dim_input=2,
                dim_output=dim_hidden,
                layer_norm=False,
                spectral_norm=spectral_norm,
            ),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder_blocks = nn.ModuleList()
        for _ in range(depth):
            self.encoder_blocks.append(
                MAB(
                    dim_q=dim_hidden,
                    dim_k=dim_hidden,
                    dim_v=dim_hidden,
                    num_heads=num_heads,
                    layer_norm=layer_norm,
                    negative_slope=negative_slope,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm,
                )
            )

    def forward(
        self, inputs: torch.Tensor, position: torch.Tensor, mask: torch.Tensor = None,
    ) -> torch.Tensor:
        position = self.position_embedding(position)
        inputs = self.input_embedding(inputs) + position
        for encoder_block in self.encoder_blocks:
            inputs = encoder_block(q=inputs, k=inputs, mask=mask)
        return inputs


class DensityAttentionNetwork(nn.Module):
    def __init__(
        self, feature_extractor: Encoder, density_estimator: nn.Module,
    ):
        super(DensityAttentionNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.density_estimator = density_estimator

    def forward(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        position: torch.Tensor,
        inputs_mask: torch.Tensor = None,
        outputs_mask: torch.Tensor = None,
    ) -> torch.distributions.Distribution:
        outputs = self.feature_extractor(
            inputs=inputs, position=position, mask=inputs_mask
        )
        outputs = outputs.reshape(-1, outputs.shape[-1])[outputs_mask]
        return self.density_estimator(outputs)


class AppendedDensityAttentionNetwork(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, density_estimator: nn.Module,
    ):
        super(AppendedDensityAttentionNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.density_estimator = density_estimator

    def forward(
        self,
        inputs: torch.Tensor,
        treatments: torch.Tensor,
        position: torch.Tensor,
        inputs_mask: torch.Tensor = None,
        outputs_mask: torch.Tensor = None,
    ) -> torch.distributions.Distribution:
        phi = self.feature_extractor(inputs=inputs, position=position, mask=inputs_mask)
        phi = phi.reshape(-1, phi.shape[-1])[outputs_mask]
        treatments = treatments.reshape(-1, treatments.shape[-1])[outputs_mask]
        phi = torch.cat([phi, treatments], dim=-1)
        return self.density_estimator(phi)


class TarAttentionNetwork(nn.Module):
    def __init__(
        self,
        feature_extractor: Encoder,
        hypotheses: nn.Module,
        density_estimator: nn.Module,
    ):
        super(TarAttentionNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.hypotheses = hypotheses
        self.density_estimator = density_estimator

    def forward(
        self,
        inputs: torch.Tensor,
        treatments: torch.Tensor,
        position: torch.Tensor,
        inputs_mask: torch.Tensor = None,
        outputs_mask: torch.Tensor = None,
    ) -> torch.distributions.Distribution:
        outputs = self.feature_extractor(
            inputs=inputs, position=position, mask=inputs_mask
        )
        outputs = outputs.reshape(-1, outputs.shape[-1])[outputs_mask]
        treatments = treatments.reshape(-1, treatments.shape[-1])[outputs_mask]
        return self.density_estimator(
            [self.hypotheses([outputs, treatments]), treatments,]
        )
