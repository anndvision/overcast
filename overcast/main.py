import ray
import click

from torch import cuda

from pathlib import Path

from overcast import workflows


@click.group(chain=True)
@click.pass_context
def cli(context):
    context.obj = {"n_gpu": cuda.device_count()}


@cli.command("train")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option(
    "--gpu-per-model",
    default=cuda.device_count(),
    type=float,
    help="number of gpus for each ensemble model, default=cuda.device_count()",
)
@click.option(
    "--seed", default=1331, type=int, help="random number generator seed, default=1331",
)
@click.pass_context
def train(
    context, job_dir, gpu_per_model, seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
        object_store_memory=8000000000,
    )
    context.obj.update(
        {
            "job_dir": job_dir,
            "gpu_per_model": gpu_per_model,
            "seed": seed,
            "mode": "train",
        }
    )


@cli.command("tune")
@click.option(
    "--job-dir",
    type=str,
    required=True,
    help="location for writing checkpoints and results",
)
@click.option(
    "--max-samples",
    default=500,
    type=int,
    help="maximum number of search space samples, default=100",
)
@click.option(
    "--gpu-per-model",
    default=cuda.device_count(),
    type=float,
    help="number of gpus for each ensemble model, default=cuda.device_count()",
)
@click.option(
    "--seed", default=1331, type=int, help="random number generator seed, default=1331",
)
@click.pass_context
def tune(
    context, job_dir, max_samples, gpu_per_model, seed,
):
    ray.init(
        num_gpus=context.obj["n_gpu"],
        dashboard_host="127.0.0.1",
        ignore_reinit_error=True,
        object_store_memory=8000000000,
    )
    context.obj.update(
        {
            "job_dir": job_dir,
            "max_samples": max_samples,
            "gpu_per_model": gpu_per_model,
            "seed": seed,
            "mode": "tune",
        }
    )


@cli.command("jasmin")
@click.pass_context
@click.option(
    "--root", type=str, required=True, help="location of dataset",
)
@click.option(
    "--covariates",
    "-c",
    type=str,
    multiple=True,
    default=["RH900", "RH850", "RH700", "LTS", "EIS", "w500", "whoi_sst",],
    help="covariate keys",
)
@click.option(
    "--treatment", "-t", type=str, default="tot_aod", help="treatment key",
)
@click.option(
    "--outcomes",
    "-o",
    type=str,
    multiple=True,
    default=["l_re", "liq_pc", "cod", "cwp"],
    help="outcome keys",
)
@click.option(
    "--num-bins",
    type=int,
    default=1,
    help="Number of bins to discretize treatment variable, default=1 (treatment is continuous)",
)
@click.option(
    "--filter-aod",
    type=bool,
    default=True,
    help="Filter out aod values less than 0.7 and greater than 1",
)
@click.option(
    "--filter-precip", type=bool, default=True, help="Filter out raining clouds",
)
@click.option(
    "--bootstrap",
    type=bool,
    default=False,
    help="bootstrap sample the training dataset, default=False",
)
def jasmin(
    context,
    root,
    covariates,
    treatment,
    outcomes,
    num_bins,
    filter_aod,
    filter_precip,
    bootstrap,
):
    outcomes = list(outcomes)
    covariates = list(covariates)
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "jasmin"
    dataset_folder = dataset_name + f"_treatment-{treatment}_covariates"
    for c in covariates:
        dataset_folder += f"-{c}"
    dataset_folder += "_outcomes"
    for o in outcomes:
        dataset_folder += f"-{o}"
    dataset_folder += f"_bins-{num_bins}"
    experiment_dir = job_dir / dataset_folder
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "root": root,
                "split": "train",
                "x_vars": covariates,
                "t_var": treatment,
                "y_vars": outcomes,
                "t_bins": num_bins,
                "filter_aod": filter_aod,
                "filter_precip": filter_precip,
                "bootstrap": bootstrap,
            },
            "ds_valid": {
                "root": root,
                "split": "valid",
                "x_vars": covariates,
                "t_var": treatment,
                "y_vars": outcomes,
                "t_bins": num_bins,
                "filter_aod": filter_aod,
                "filter_precip": filter_precip,
                "bootstrap": False,
            },
            "ds_test": {
                "root": root,
                "split": "test",
                "x_vars": covariates,
                "t_var": treatment,
                "y_vars": outcomes,
                "t_bins": num_bins,
                "filter_aod": filter_aod,
                "filter_precip": filter_precip,
                "bootstrap": False,
            },
        }
    )


@cli.command("jasmin-daily")
@click.pass_context
@click.option(
    "--root", type=str, required=True, help="location of dataset",
)
@click.option(
    "--covariates",
    "-c",
    type=str,
    multiple=True,
    default=["RH900", "RH850", "RH700", "LTS", "EIS", "w500", "whoi_sst",],
    help="covariate keys",
)
@click.option(
    "--treatment", "-t", type=str, default="tot_aod", help="treatment key",
)
@click.option(
    "--outcomes",
    "-o",
    type=str,
    multiple=True,
    default=["l_re", "liq_pc", "cod", "cwp"],
    help="outcome keys",
)
@click.option(
    "--num-bins",
    type=int,
    default=1,
    help="Number of bins to discretize treatment variable, default=2",
)
@click.option(
    "--filter-aod",
    type=bool,
    default=True,
    help="Filter out aod values less than 0.7 and greater than 1",
)
@click.option(
    "--filter-precip", type=bool, default=True, help="Filter out raining clouds",
)
@click.option(
    "--bootstrap",
    type=bool,
    default=False,
    help="bootstrap sample the training dataset, default=False",
)
def jasmin_daily(
    context,
    root,
    covariates,
    treatment,
    outcomes,
    num_bins,
    filter_aod,
    filter_precip,
    bootstrap,
):
    outcomes = list(outcomes)
    covariates = list(covariates)
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "jasmin-daily"
    dataset_folder = dataset_name + f"_treatment-{treatment}_covariates"
    for c in covariates:
        dataset_folder += f"-{c}"
    dataset_folder += "_outcomes"
    for o in outcomes:
        dataset_folder += f"-{o}"
    dataset_folder += f"_bins-{num_bins}"
    dataset_folder += f"_bs-{bootstrap}"
    experiment_dir = job_dir / dataset_folder
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "root": root,
                "split": "train",
                "x_vars": covariates,
                "t_var": treatment,
                "y_vars": outcomes,
                "t_bins": num_bins,
                "filter_aod": filter_aod,
                "filter_precip": filter_precip,
                "pad": True,
                "bootstrap": bootstrap,
            },
            "ds_valid": {
                "root": root,
                "split": "valid",
                "x_vars": covariates,
                "t_var": treatment,
                "y_vars": outcomes,
                "t_bins": num_bins,
                "filter_aod": filter_aod,
                "filter_precip": filter_precip,
                "pad": True,
                "bootstrap": False,
            },
            "ds_test": {
                "root": root,
                "split": "test",
                "x_vars": covariates,
                "t_var": treatment,
                "y_vars": outcomes,
                "t_bins": num_bins,
                "filter_aod": filter_aod,
                "filter_precip": filter_precip,
                "pad": False,
                "bootstrap": False,
            },
        }
    )


@cli.command("dose-response")
@click.pass_context
@click.option(
    "--num-examples", type=int, default=10000, help="dataset size, default=10000",
)
@click.option(
    "--gamma-t", type=float, default=0.3, help="default=0.3",
)
@click.option(
    "--gamma-y", type=float, default=0.5, help="default=0.5",
)
@click.option(
    "--bootstrap",
    type=bool,
    default=False,
    help="bootstrap sample the training dataset, default=False",
)
@click.option(
    "--seed", type=int, default=0, help="default=0",
)
def dose_response(
    context, num_examples, gamma_t, gamma_y, bootstrap, seed,
):
    job_dir = Path(context.obj.get("job_dir"))
    dataset_name = "dose-response"
    dataset_folder = (
        dataset_name + f"_n-{num_examples}_gt-{gamma_t}_gy-{gamma_y}_seed-{seed}"
    )
    experiment_dir = job_dir / dataset_folder
    context.obj.update(
        {
            "dataset_name": dataset_name,
            "experiment_dir": str(experiment_dir),
            "ds_train": {
                "num_examples": num_examples,
                "gamma_t": gamma_t,
                "gamma_y": gamma_y,
                "sigma_y": 0.2,
                "domain": 2.0,
                "p_u": "bernoulli",
                "bootstrap": bootstrap,
                "seed": seed,
                "split": None,
            },
            "ds_valid": {
                "num_examples": num_examples // 10,
                "gamma_t": gamma_t,
                "gamma_y": gamma_y,
                "sigma_y": 0.2,
                "domain": 2.0,
                "p_u": "bernoulli",
                "bootstrap": False,
                "seed": seed + 1,
                "split": None,
            },
            "ds_test": {
                "num_examples": num_examples // 10,
                "gamma_t": gamma_t,
                "gamma_y": gamma_y,
                "sigma_y": 0.2,
                "domain": 2.0,
                "p_u": "bernoulli",
                "bootstrap": False,
                "seed": seed + 2,
                "split": None,
            },
        }
    )


@cli.command("discrete-treatment-nn")
@click.pass_context
@click.option("--dim-hidden", default=512, type=int, help="num neurons, default=512")
@click.option(
    "--num-components", default=23, type=int, help="num mixture components, default=512"
)
@click.option(
    "--depth", default=3, type=int, help="depth of feature extractor, default=3"
)
@click.option(
    "--negative-slope",
    default=0.01,
    type=float,
    help="negative slope of leaky relu, default=0.01",
)
@click.option(
    "--beta", default=0.0, type=float, help="double robust regularizer, default=0.0"
)
@click.option(
    "--layer-norm", default=False, type=bool, help="use layer norm, default=False"
)
@click.option(
    "--dropout-rate", default=0.2, type=float, help="dropout rate, default=0.2"
)
@click.option(
    "--spectral-norm",
    default=0.0,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=1e-4,
    type=float,
    help="learning rate for gradient descent, default=1e-4",
)
@click.option(
    "--batch-size",
    default=2048,
    type=int,
    help="number of examples to read during each training step, default=4096",
)
@click.option(
    "--epochs", type=int, default=400, help="number of training epochs, default=400"
)
@click.option(
    "--ensemble-size",
    type=int,
    default=10,
    help="number of models in ensemble, default=10",
)
def discrete_treatment_nn(
    context,
    dim_hidden,
    num_components,
    depth,
    negative_slope,
    beta,
    layer_norm,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
    ensemble_size,
):
    context.obj.update(
        {
            "experiment_dir": str(
                Path(context.obj["experiment_dir"]) / "discrete_treatment_nn"
            )
        }
    )
    if context.obj["mode"] == "tune":
        context.obj.update(
            {"epochs": epochs, "ensemble_size": ensemble_size,}
        )
        workflows.neural_network.discrete_treatment.tune_func(config=context.obj)
    else:
        context.obj.update(
            {
                "dim_hidden": dim_hidden,
                "depth": depth,
                "num_components": num_components,
                "negative_slope": negative_slope,
                "beta": beta,
                "layer_norm": layer_norm,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "ensemble_size": ensemble_size,
            }
        )

        if context.obj["mode"] == "train":

            @ray.remote(num_gpus=context.obj.get("gpu_per_model"),)
            def trainer(**kwargs):
                func = workflows.neural_network.discrete_treatment.train(**kwargs)
                return func

            results = []
            for ensemble_id in range(ensemble_size):
                results.append(
                    trainer.remote(
                        config=context.obj,
                        experiment_dir=context.obj.get("experiment_dir"),
                        ensemble_id=ensemble_id,
                    )
                )
            ray.get(results)


@cli.command("appended-treatment-nn")
@click.pass_context
@click.option("--dim-hidden", default=256, type=int, help="num neurons, default=512")
@click.option(
    "--depth", default=3, type=int, help="depth of feature extractor, default=3"
)
@click.option(
    "--num-components-outcome",
    default=24,
    type=int,
    help="num mixture components for outcome density estimator, default=22",
)
@click.option(
    "--num-components-treatment",
    default=24,
    type=int,
    help="num mixture components for treatment density estimator, default=14",
)
@click.option(
    "--negative-slope",
    default=0.04,
    type=float,
    help="negative slope of leaky relu, default=0.01",
)
@click.option(
    "--beta", default=0.0, type=float, help="double robust regularizer, default=0.0"
)
@click.option(
    "--layer-norm", default=False, type=bool, help="use layer norm, default=False"
)
@click.option(
    "--dropout-rate", default=0.2, type=float, help="dropout rate, default=0.25"
)
@click.option(
    "--spectral-norm",
    default=0.0,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=1e-4,
    type=float,
    help="learning rate for gradient descent, default=1e-4",
)
@click.option(
    "--batch-size",
    default=2048,
    type=int,
    help="number of examples to read during each training step, default=2048",
)
@click.option(
    "--epochs", type=int, default=300, help="number of training epochs, default=200"
)
@click.option(
    "--ensemble-size",
    type=int,
    default=10,
    help="number of models in ensemble, default=10",
)
def appended_treatment_nn(
    context,
    dim_hidden,
    depth,
    num_components_outcome,
    num_components_treatment,
    negative_slope,
    beta,
    layer_norm,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
    ensemble_size,
):
    context.obj.update(
        {
            "experiment_dir": str(
                Path(context.obj["experiment_dir"]) / "appended-treatment-nn"
            )
        }
    )
    if context.obj["mode"] == "tune":
        context.obj.update(
            {"epochs": 9, "ensemble_size": ensemble_size,}
        )
        workflows.neural_network.appended_treatment.tune_func(config=context.obj)
    else:
        context.obj.update(
            {
                "dim_hidden": dim_hidden,
                "depth": depth,
                "num_components_outcome": num_components_outcome,
                "num_components_treatment": num_components_treatment,
                "negative_slope": negative_slope,
                "beta": beta,
                "layer_norm": layer_norm,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "ensemble_size": ensemble_size,
            }
        )

        if context.obj["mode"] == "train":

            @ray.remote(num_gpus=context.obj.get("gpu_per_model"))
            def trainer(**kwargs):
                func = workflows.neural_network.appended_treatment.train(**kwargs)
                return func

            results = []
            for ensemble_id in range(ensemble_size):
                results.append(
                    trainer.remote(
                        config=context.obj,
                        experiment_dir=context.obj.get("experiment_dir"),
                        ensemble_id=ensemble_id,
                    )
                )
            ray.get(results)


@cli.command("appended-treatment-transformer")
@click.pass_context
@click.option("--dim-hidden", default=256, type=int, help="num neurons, default=512")
@click.option(
    "--depth", default=3, type=int, help="depth of feature extractor, default=3"
)
@click.option(
    "--num-components-outcome",
    default=24,
    type=int,
    help="num mixture components for outcome density estimator, default=22",
)
@click.option(
    "--num-components-treatment",
    default=24,
    type=int,
    help="num mixture components for treatment density estimator, default=14",
)
@click.option(
    "--num-heads", default=4, type=int, help="multi-head attention parameter, default=2"
)
@click.option(
    "--negative-slope",
    default=0.01,
    type=float,
    help="negative slope of leaky relu, default=0.01",
)
@click.option(
    "--beta", default=0.0, type=float, help="double robust regularizer, default=0.0"
)
@click.option(
    "--layer-norm", default=False, type=bool, help="use layer norm, default=False"
)
@click.option(
    "--dropout-rate", default=0.5, type=float, help="dropout rate, default=0.25"
)
@click.option(
    "--spectral-norm",
    default=0.0,
    type=float,
    help="Spectral normalization coefficient. If 0.0 do not use spectral norm, default=0.0",
)
@click.option(
    "--learning-rate",
    default=2e-4,
    type=float,
    help="learning rate for gradient descent, default=1e-4",
)
@click.option(
    "--batch-size",
    default=32,
    type=int,
    help="number of examples to read during each training step, default=2048",
)
@click.option(
    "--epochs", type=int, default=500, help="number of training epochs, default=500"
)
@click.option(
    "--ensemble-size",
    type=int,
    default=10,
    help="number of models in ensemble, default=10",
)
def appended_treatment_an(
    context,
    dim_hidden,
    depth,
    num_components_outcome,
    num_components_treatment,
    num_heads,
    negative_slope,
    beta,
    layer_norm,
    dropout_rate,
    spectral_norm,
    learning_rate,
    batch_size,
    epochs,
    ensemble_size,
):
    context.obj.update(
        {
            "experiment_dir": str(
                Path(context.obj["experiment_dir"]) / "appended-treatment-transformer"
            )
        }
    )
    if context.obj["mode"] == "tune":
        context.obj.update(
            {"epochs": epochs, "ensemble_size": ensemble_size,}
        )
        workflows.transformer.appended_treatment.tune_func(config=context.obj)
    else:
        context.obj.update(
            {
                "num_components_outcome": num_components_outcome,
                "num_components_treatment": num_components_treatment,
                "dim_hidden": dim_hidden,
                "depth": depth,
                "num_heads": num_heads,
                "negative_slope": negative_slope,
                "beta": beta,
                "layer_norm": layer_norm,
                "dropout_rate": dropout_rate,
                "spectral_norm": spectral_norm,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "ensemble_size": ensemble_size,
            }
        )

        if context.obj["mode"] == "train":

            @ray.remote(num_gpus=context.obj.get("gpu_per_model"),)
            def trainer(**kwargs):
                func = workflows.transformer.appended_treatment.train(**kwargs)
                return func

            results = []
            for ensemble_id in range(ensemble_size):
                results.append(
                    trainer.remote(
                        config=context.obj,
                        experiment_dir=context.obj.get("experiment_dir"),
                        ensemble_id=ensemble_id,
                    )
                )
            ray.get(results)
