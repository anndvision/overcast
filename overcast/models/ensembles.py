import numpy as np


def predict_mean(ensemble, dataset, batch_size=None):
    mean_ensemble = []
    for model in ensemble:
        mean_ensemble.append(
            np.expand_dims(
                model.predict_mean(dataset=dataset, batch_size=batch_size), axis=0
            )
        )
    return np.concatenate(mean_ensemble, axis=0)


def sample_pos(ensemble, dataset, treatments, num_samples=100, batch_size=None):
    pos_ensemble = []
    for model in ensemble:
        pos = []
        for treatment in treatments:
            po = model.sample_po(
                dataset, treatment, num_samples=num_samples, batch_size=batch_size
            )
            pos.append(np.expand_dims(po.transpose(2, 0, 1), -1))
        pos = np.concatenate(pos, axis=-1)
        pos_ensemble.append(pos)
    return np.concatenate(pos_ensemble, axis=1)


def predict_capos(ensemble, dataset, treatments, batch_size=None):
    capos_ensemble = []
    for model in ensemble:
        capos = []
        for treatment in treatments:
            capo = model.predict_capo(dataset, treatment, batch_size=batch_size)
            capos.append(np.expand_dims(capo.transpose(), -1))
        capos = np.concatenate(capos, axis=-1)
        capos_ensemble.append(np.expand_dims(capos, 1))
    return np.concatenate(capos_ensemble, axis=1)


def predict_intervals(
    ensemble, dataset, treatments, log_lambda, num_samples=100, batch_size=None
):
    upper_ensemble = []
    lower_ensemble = []
    for model in ensemble:
        upper, lower = [], []
        for treatment in treatments:
            intervals = model.predict_capo_interval(
                dataset,
                treatment,
                log_lambda=log_lambda,
                num_samples=num_samples,
                batch_size=batch_size,
            )
            lower.append(np.expand_dims(intervals[0].transpose(), -1))
            upper.append(np.expand_dims(intervals[1].transpose(), -1))
        lower_ensemble.append(np.expand_dims(np.concatenate(lower, axis=-1), 1))
        upper_ensemble.append(np.expand_dims(np.concatenate(upper, axis=-1), 1))
    return (
        np.concatenate(lower_ensemble, axis=1),
        np.concatenate(upper_ensemble, axis=1),
    )
