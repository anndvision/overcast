eps = 1e-7


def alpha_func(p, _lambda):
    return 1 / (_lambda * p) + 1 - 1 / (_lambda)


def beta_func(p, _lambda):
    return _lambda / (p) + 1 - _lambda
