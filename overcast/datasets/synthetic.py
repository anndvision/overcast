import numpy as np

import pandas as pd

from scipy import stats

from torch.utils import data


class DoseResponse(data.Dataset):
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
        super(DoseResponse, self).__init__()
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

        mu_y = f_mu_dr(x=self.data, t=self.treatments, u=self.u, theta=gamma_y).astype(
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

        if bootstrap:
            sample_index = np.random.choice(
                np.arange(len(self.data)), size=len(self.data)
            )
            self.data = self.data[sample_index]
            self.treatments = self.treatments[sample_index]
            self.targets = self.targets[sample_index]
            self.u = self.u[sample_index]

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
            mu_y = f_mu_dr(x=data, t=treatment, u=self.u, theta=self.gamma_y)
            mus.append(mu_y.mean())
        return np.asarray(mus)

    def conditional_dr_curve(self, treatments):
        data = self.data
        mus = []
        for treatment in treatments:
            treatment = treatment * np.ones_like(self.treatments)
            mu_y = 0.5 * f_mu_dr(
                x=data, t=treatment, u=np.ones_like(self.u), theta=self.gamma_y
            ) + 0.5 * f_mu_dr(
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


class CATE(data.Dataset):
    def __init__(
        self,
        num_examples,
        lambda_star,
        p_u="bernoulli",
        theta=4.0,
        beta=0.75,
        sigma_y=1.0,
        domain=2,
        bootstrap=False,
        seed=1331,
        split=None,
    ):
        super(CATE, self).__init__()
        rng = np.random.RandomState(seed=seed)

        self.num_examples = num_examples

        self.data = rng.uniform(-domain, domain, size=(num_examples, 1)).astype(
            "float32"
        )

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

        self.pi = complete_propensity(
            x=self.data, u=self.u, _lambda=lambda_star, beta=beta
        ).astype("float32")
        self.treatments = rng.binomial(1, self.pi).astype("float32")
        self.treatments = np.hstack([1 - self.treatments, self.treatments])

        self.mu0 = f_mu_cate(x=self.data, t=0.0, u=self.u, theta=theta).astype(
            "float32"
        )
        self.mu1 = f_mu_cate(x=self.data, t=1.0, u=self.u, theta=theta).astype(
            "float32"
        )
        eps = (sigma_y * rng.normal(size=self.data.shape)).astype("float32")
        self.y0 = self.mu0 + eps
        self.y1 = self.mu1 + eps
        self.targets = (
            self.treatments[:, 1:] * self.y1 + self.treatments[:, :1] * self.y0
        )
        self.tau = self.mu1 - self.mu0

        self.data_xfm = None
        self.targets_xfm = None
        self.treatments_xfm = None

        self.dim_input = 1
        self.dim_targets = 1
        self.dim_treatments = 2

        self.data_names = ["x1"]
        self.target_names = ["y"]
        self.treatment_names = ["t"]

        if bootstrap:
            sample_index = np.random.choice(
                np.arange(len(self.data)), size=len(self.data)
            )
            self.data = self.data[sample_index]
            self.treatments = self.treatments[sample_index]
            self.targets = self.targets[sample_index]
            self.u = self.u[sample_index]

    @property
    def data_frame(self) -> pd.DataFrame:
        data = np.hstack([self.data, self.treatments, self.targets,],)
        return pd.DataFrame(
            data=data,
            columns=self.data_names + self.treatment_names + self.target_names,
        )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> data.dataset.T_co:
        return self.data[index], self.treatments[index], self.targets[index]


def f_mu_dr(x, t, u, theta=4.0):
    mu = t + x * np.exp(-x * t) - (theta * (u - 0.5)) * (0.5 * x + 1)
    return mu


def f_mu_cate(x, t, u, theta=4.0):
    mu = (
        (2 * t - 1) * x
        + (2.0 * t - 1)
        - 2 * np.sin((4 * t - 2) * x)
        - (theta * u - 2) * (1 + 0.5 * x)
    )
    return mu


def f_t(x, u, lambda_star=0.02):
    mu = np.exp(x - 1) + (1 - lambda_star) * (2 * u - 1)
    return mu


def complete_propensity(x, u, _lambda, beta=0.75):
    logit = beta * x + 0.5
    nominal = (1 + np.exp(-logit)) ** -1
    alpha = alpha_fn(nominal, _lambda)
    beta = beta_fn(nominal, _lambda)
    return (u / alpha) + ((1 - u) / beta)


def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_ ** -1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_
