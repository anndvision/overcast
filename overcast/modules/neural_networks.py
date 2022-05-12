import torch
from torch import nn


class NeuralDensityNetwork(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, density_estimator: nn.Module,
    ):
        super(NeuralDensityNetwork, self).__init__()
        self.op = nn.Sequential(feature_extractor, density_estimator,)

    def forward(self, inputs):
        return self.op(inputs)


class AppendedDensityNetwork(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, density_estimator: nn.Module,
    ):
        super(AppendedDensityNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.density_estimator = density_estimator

    def forward(self, inputs, treatments):
        phi = self.feature_extractor(inputs)
        phi = torch.cat([phi, treatments], dim=-1)
        return self.density_estimator(phi)


class TarNet(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        hypotheses: nn.Module,
        density_estimator: nn.Module,
    ):
        super(TarNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.hypotheses = hypotheses
        self.density_estimator = density_estimator

    def forward(self, inputs, treatments):
        phi = self.feature_extractor(inputs)
        phi = self.hypotheses([phi, treatments])
        return self.density_estimator([phi, treatments])
