import torch
from torch import distributions, nn

from overcast.modules import group
from overcast.modules import conditional


class Normal(nn.Module):
    def __init__(
        self, dim_input, dim_output,
    ):
        super(Normal, self).__init__()
        self.mu = nn.Linear(in_features=dim_input, out_features=dim_output, bias=True,)
        sigma = nn.Linear(in_features=dim_input, out_features=dim_output, bias=True,)
        self.sigma = nn.Sequential(sigma, nn.Softplus())

    def forward(self, inputs):
        return distributions.Normal(
            loc=self.mu(inputs), scale=self.sigma(inputs) + 1e-7
        )


class GroupNormal(nn.Module):
    def __init__(
        self, dim_input, dim_output, groups,
    ):
        super(GroupNormal, self).__init__()
        self.mu = group.GroupLinear(
            dim_input=dim_input, dim_output=dim_output, num_groups=groups, bias=True
        )
        self.sigma = nn.Sequential(
            group.GroupLinear(
                dim_input=dim_input,
                dim_output=dim_output,
                num_groups=groups,
                bias=True,
            ),
            nn.Softplus(),
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        loc = self.mu(inputs)
        scale = self.sigma(inputs)
        return distributions.Normal(loc=loc, scale=scale + 1e-7)


class MixtureSameFamily(distributions.MixtureSameFamily):
    def log_prob(self, inputs):
        loss = torch.exp(self.component_distribution.log_prob(inputs.unsqueeze(1)))
        loss = torch.sum(loss * self.mixture_distribution.probs, dim=1)
        return torch.log(loss + 1e-7)

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.base_dist.cdf(x).squeeze(-1)
        mix_prob = self.mixture_distribution.probs
        return torch.sum(cdf_x * mix_prob, dim=-1)


class GMM(nn.Module):
    def __init__(
        self, num_components, dim_input, dim_output,
    ):
        super(GMM, self).__init__()
        self.mu = nn.Linear(
            in_features=dim_input, out_features=num_components * dim_output, bias=True,
        )
        sigma = nn.Linear(
            in_features=dim_input, out_features=num_components * dim_output, bias=True,
        )
        self.pi = nn.Linear(
            in_features=dim_input, out_features=num_components, bias=True,
        )
        self.sigma = nn.Sequential(sigma, nn.Softplus())
        self.num_components = num_components
        self.dim_output = dim_output

    def forward(self, inputs):
        loc = self.mu(inputs).reshape(-1, self.num_components, self.dim_output)
        scale = (
            self.sigma(inputs).reshape(-1, self.num_components, self.dim_output) + 1e-7
        )
        return MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=self.pi(inputs)),
            component_distribution=distributions.Independent(
                base_distribution=distributions.Normal(loc=loc, scale=scale,),
                reinterpreted_batch_ndims=1,
            ),
        )


class GroupGMM(nn.Module):
    def __init__(self, num_components, dim_input, dim_output, groups):
        super(GroupGMM, self).__init__()
        self.mu = group.GroupLinear(
            dim_input=dim_input,
            dim_output=dim_output * num_components,
            num_groups=groups,
            bias=True,
        )
        self.sigma = nn.Sequential(
            group.GroupLinear(
                dim_input=dim_input,
                dim_output=dim_output * num_components,
                num_groups=groups,
                bias=True,
            ),
            nn.Softplus(),
        )
        self.pi = group.GroupLinear(
            dim_input=dim_input,
            dim_output=num_components,
            num_groups=groups,
            bias=True,
        )
        self.dim_output = dim_output
        self.num_components = num_components

    def forward(self, inputs):

        logits = self.pi(inputs)
        loc = self.mu(inputs)
        scale = self.sigma(inputs) + 1e-7

        component_distribution = distributions.Independent(
            distributions.Normal(
                loc=loc.reshape(-1, self.num_components, self.dim_output),
                scale=scale.reshape(-1, self.num_components, self.dim_output),
            ),
            reinterpreted_batch_ndims=1,
        )
        return MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=logits),
            component_distribution=component_distribution,
        )


class ConditionalGMM(nn.Module):
    def __init__(
        self,
        num_components,
        dim_input,
        dim_condition,
        dim_output,
        num_basis,
        dim_basis,
    ):
        super(ConditionalGMM, self).__init__()
        self.mu = conditional.ConditionalLinear(
            dim_input=dim_input,
            condition_features=dim_condition,
            dim_output=dim_output * num_components,
            num_basis=num_basis,
            basis_features=dim_basis,
            bias=True,
        )
        self.sigma = nn.Sequential(
            conditional.ConditionalLinear(
                dim_input=dim_input,
                condition_features=dim_condition,
                dim_output=dim_output * num_components,
                num_basis=num_basis,
                basis_features=dim_basis,
                bias=True,
            ),
            nn.Softplus(),
        )
        self.pi = conditional.ConditionalLinear(
            dim_input=dim_input,
            condition_features=dim_condition,
            dim_output=num_components,
            num_basis=num_basis,
            basis_features=dim_basis,
            bias=True,
        )
        self.dim_output = dim_output
        self.num_components = num_components

    def forward(self, inputs):

        logits = self.pi(inputs)
        loc = self.mu(inputs)
        scale = self.sigma(inputs) + 1e-7

        component_distribution = distributions.Independent(
            distributions.Normal(
                loc=loc.reshape(-1, self.num_components, self.dim_output),
                scale=scale.reshape(-1, self.num_components, self.dim_output),
            ),
            reinterpreted_batch_ndims=1,
        )
        return MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=logits),
            component_distribution=component_distribution,
        )


class Categorical(nn.Module):
    def __init__(
        self, dim_input, dim_output,
    ):
        super(Categorical, self).__init__()
        self.logits = nn.Linear(
            in_features=dim_input, out_features=dim_output, bias=True,
        )
        self.distribution = (
            distributions.Bernoulli
            if dim_output == 1
            else distributions.OneHotCategorical
        )

    def forward(self, inputs):
        logits = self.logits(inputs)
        return self.distribution(logits=logits)
