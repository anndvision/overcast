import numpy as np

import pandas as pd

from scipy import stats

from torch.utils import data


class Synthetic(data.Dataset):
    def __init__(
        self,
        num_examples,
        gamma_t=0.0,
        gamma_y=0.0,
        sigma_y=0.2,
        domain=2.0,
        p_u="bernoulli",
        bootstrap=False,
        seed=1331,
        split=None,
    ):
        super(Synthetic, self).__init__()
        rng = np.random.RandomState(seed=seed)

        self.num_examples = num_examples
        self.gamma_y = gamma_y
        self.gamma_t = gamma_t

        self.data = rng.uniform(0.1, domain, size=(num_examples, 1)).astype("float32")

        if p_u == "bernoulli":
            self.u = rng.binomial(1, 0.5, size=(num_examples, 1)).astype("float32")
        elif p_u == "uniform":
            self.u = rng.uniform(size=(num_examples, 1)).astype("float32")
        elif p_u == "beta_bi":
            self.u = rng.beta(0.5, 0.5, size=(num_examples, 1)).astype("float32")
        elif p_u == "beta_uni":
            self.u = rng.beta(2, 5, size=(num_examples, 1)).astype("float32")
        else:
            raise NotImplementedError(f"{p_u} is not a supported distribution")

        self.treatments = stats.betabinom.rvs(
            n=100, a=self.data + gamma_t * self.u, b=1.0, random_state=seed
        ).astype("float32")
        n = 0.5 * stats.betabinom.pmf(
            self.treatments, n=100, a=self.data, b=1.0
        ) + 0.5 * stats.betabinom.pmf(
            self.treatments, n=100, a=self.data + gamma_t, b=1.0
        )
        d = stats.betabinom.pmf(
            self.treatments, n=100, a=self.data + gamma_t * self.u, b=1.0
        )
        self.lambda_star = n / d
        self.r = (n / (1 - n)) / (d / (1 - d))
        self.treatments = self.treatments / 100

        mu_y = f_mu(x=self.data, t=self.treatments, u=self.u, theta=gamma_y).astype(
            "float32"
        )
        eps = (sigma_y * rng.normal(size=self.data.shape)).astype("float32")
        self.targets = mu_y + eps

        self.data_xfm = None
        self.targets_xfm = None
        self.treatments_xfm = None

        self.dim_input = 1
        self.dim_targets = 1
        self.dim_treatments = 1

        self.data_names = ["x1"]
        self.target_names = ["y"]
        self.treatment_names = ["t"]

        self.sample_index = np.arange(len(self.data))
        if bootstrap:
            self.sample_index = np.random.choice(self.sample_index, size=len(self.data))
            self.data = self.data[self.sample_index]
            self.treatments = self.treatments[self.sample_index]
            self.targets = self.targets[self.sample_index]
            self.u = self.u[self.sample_index]

    @property
    def data_frame(self) -> pd.DataFrame:
        data = np.hstack([self.data, self.treatments, self.targets,],)
        return pd.DataFrame(
            data=data,
            columns=self.data_names + self.treatment_names + self.target_names,
        )

    def dr_curve(self, treatments):
        data = self.data
        mus = []
        for treatment in treatments:
            treatment = treatment * np.ones_like(self.treatments)
            mu_y = f_mu(x=data, t=treatment, u=self.u, theta=self.gamma_y)
            mus.append(mu_y.mean())
        return np.asarray(mus)

    def conditional_dr_curve(self, treatments):
        data = self.data
        mus = []
        for treatment in treatments:
            treatment = treatment * np.ones_like(self.treatments)
            mu_y = 0.5 * f_mu(
                x=data, t=treatment, u=np.ones_like(self.u), theta=self.gamma_y
            ) + 0.5 * f_mu(
                x=data, t=treatment, u=np.zeros_like(self.u), theta=self.gamma_y
            )
            mus.append(mu_y)
        return np.asarray(mus)

    def conditional_lambda(self, treatments):
        lambdas = []
        for treatment in treatments:
            treatment = treatment * 100 * np.ones_like(self.treatments)
            n = 0.5 * stats.betabinom.pmf(
                treatment, n=100, a=self.data, b=1.0
            ) + 0.5 * stats.betabinom.pmf(
                treatment, n=100, a=self.data + self.gamma_t, b=1.0
            )
            d = stats.betabinom.pmf(
                treatment, n=100, a=self.data + self.gamma_t * self.u, b=1.0
            )
            lambdas.append(n / (d + 1e-7))
        return np.asarray(lambdas)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> data.dataset.T_co:
        return self.data[index], self.treatments[index], self.targets[index]


def complete_propensity(x, u, gamma, beta=0.75):
    logit = beta * x + 0.5
    nominal = (1 + np.exp(-logit)) ** -1
    alpha = alpha_fn(nominal, gamma)
    beta = beta_fn(nominal, gamma)
    return (u / alpha) + ((1 - u) / beta)


def f_mu(x, t, u, theta=4.0):
    mu = t + x * np.exp(-x * t) - (theta * (u - 0.5)) * (0.5 * x + 1)
    return mu


def f_t(x, u, lambda_star=0.02):
    mu = np.exp(x - 1) + (1 - lambda_star) * (2 * u - 1)
    return mu


def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_ ** -1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_
