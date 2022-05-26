import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rc = {
    "figure.constrained_layout.use": True,
    "figure.facecolor": "white",
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "legend.frameon": True,
    "figure.figsize": (6, 6),
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
}
_ = sns.set(style="whitegrid", palette="colorblind", rc=rc)


def apo_curve(
    dataset,
    treatments,
    capos,
    upper_capos,
    lower_capos,
    lambda_,
    alpha=0.05,
    figsize=(6, 6),
    linewidth=4.0,
    xlim=None,
    ylim=None,
    save_path=None,
):
    _ = plt.figure(figsize=figsize)
    _ = sns.lineplot(
        x=treatments,
        y=dataset.dr_curve(treatments),
        linewidth=linewidth,
        color="C3",
        label="$\mu(\mathrm{t})$",
    )
    _ = plt.plot(
        treatments,
        capos.mean((0, 1)),
        color="C3",
        linestyle="--",
        label=r"$\widetilde{\mu}(\mathrm{t}; \theta)$",
    )
    _ = plt.fill_between(
        x=treatments,
        y1=np.quantile(upper_capos.mean(1), 1 - alpha / 2, axis=0),
        y2=np.quantile(lower_capos.mean(1), alpha / 2, axis=0),
        color="C3",
        alpha=0.2,
        label=r"$\mathrm{CI}_b(\mu(\mathrm{t};\Lambda, \alpha))$",
    )
    _ = plt.xlabel("$\mathrm{t}$")
    _ = plt.ylabel("$\mu(\mathrm{t})$")
    _ = plt.legend(loc="upper left")
    _ = plt.legend(
        title=f"$\Lambda=$" + f"{lambda_:.01f}, " + r"$\alpha=$" + f"{alpha}",
        loc="upper left",
    )
    _ = plt.xlim(xlim)
    _ = plt.ylim(ylim)
    if save_path is not None:
        _ = plt.savefig(save_path / f"apo-lamb-{lambda_:.01f}.png", dpi=150)


def capo_curves(
    dataset,
    treatments,
    capos,
    upper_capos,
    lower_capos,
    treatment_indices,
    lambda_,
    alpha=0.05,
    figsize=(6, 6),
    linewidth=4.0,
    xlim=None,
    ylim=None,
    plot_legend=False,
    save_path=None,
):
    idx_x = np.argsort(dataset.data.ravel())[::10]
    _, ax = plt.subplots(figsize=figsize)
    cdr = dataset.conditional_dr_curve(treatments)
    c_lambda = dataset.conditional_lambda(treatments)
    for i, idx_t in enumerate(treatment_indices):
        idx_coverage = (c_lambda[idx_t, idx_x].ravel() < lambda_) & (
            c_lambda[idx_t, idx_x].ravel() >= 1
        ) | (c_lambda[idx_t, idx_x].ravel() > 1 / lambda_) & (
            c_lambda[idx_t, idx_x].ravel() <= 1
        )
        _ = plt.plot(
            dataset.data[idx_x].ravel()[~idx_coverage],
            cdr[idx_t][idx_x].ravel()[~idx_coverage],
            linewidth=linewidth / 2,
            label="$\mu(\mathrm{x},$" + f"{treatments[idx_t]:.01f}" + "$)$",
            color=f"C{i}",
        )
        _ = plt.plot(
            dataset.data[idx_x].ravel()[idx_coverage],
            cdr[idx_t][idx_x].ravel()[idx_coverage],
            linewidth=linewidth,
            label="$\Lambda \leq \lambda_{<1}^{*} \mid \lambda_{>1}^{*} \leq \Lambda$",
            color=f"C{i}",
        )
        _ = plt.plot(
            dataset.data[idx_x].ravel(),
            capos[:, idx_x, idx_t].mean(0),
            color=f"C{i}",
            linestyle="--",
            label="$\widetilde{\mu}(\mathrm{x},$"
            + f"{treatments[idx_t]:.01f}"
            + r"$;\theta)$",
        )
        _ = plt.fill_between(
            x=dataset.data[idx_x].ravel(),
            y1=np.quantile(upper_capos[:, idx_x, idx_t], 1 - alpha / 2, axis=0),
            y2=np.quantile(lower_capos[:, idx_x, idx_t], alpha / 2, axis=0),
            color=f"C{i}",
            alpha=0.2,
            label=r"$\mathrm{CI}_b(\mu(\mathrm{x},$"
            + f"{treatments[idx_t]:.01f}"
            + r"$;\Lambda, \alpha))$",
        )
        _ = plt.xlabel("$\mathrm{x}$")
        _ = plt.ylabel("$\mu(\mathrm{x}, \mathrm{T})$")
        if plot_legend:
            _ = ax.legend(
                title=f"$\Lambda=$" + f"{lambda_:.01f}, " + r"$\alpha=$" + f"{alpha}",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
        _ = plt.xlim(xlim)
        _ = plt.ylim(ylim)
        if save_path is not None:
            _ = plt.savefig(save_path / f"capo-lamb-{lambda_:.01f}.png", dpi=150)


def apo_ignorance(
    dataset,
    treatments,
    apos,
    upper_apos,
    lower_apos,
    treatment_indices,
    lambda_,
    alpha=0.05,
    figsize=(6, 6),
    linewidth=4.0,
    xlim=None,
    ylim=None,
    plot_legend=False,
    save_path=None,
):
    print("hello")
