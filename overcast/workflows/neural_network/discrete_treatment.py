import json

from pathlib import Path

from overcast import models
from overcast import datasets

from ray import tune
from ray.tune import schedulers
from ray.tune.suggest import bohb


def train(
    config, experiment_dir, ensemble_id,
):
    dataset_name = config.get("dataset_name")
    train_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    valid_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

    dim_hidden = config.get("dim_hidden")
    num_components = config.get("num_components")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    beta = config.get("beta")
    layer_norm = config.get("layer_norm")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")

    experiment_dir = (
        Path(experiment_dir)
        / f"dh-{dim_hidden}_nc-{num_components}_dp-{depth}_ns-{negative_slope}_beta-{beta}_ln-{layer_norm}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config_path = experiment_dir / "config.json"
    with config_path.open(mode="w") as cp:
        json.dump(config, cp)

    out_dir = experiment_dir / "checkpoints" / f"model-{ensemble_id}" / "mu"
    if not (out_dir / "best_checkpoint.pt").exists():
        outcome_model = models.DiscreteTreatmentNeuralNetwork(
            job_dir=out_dir,
            architecture="resnet",
            dim_input=train_dataset.dim_input,
            dim_treatment=train_dataset.dim_treatments,
            dim_output=train_dataset.dim_targets,
            dim_hidden=dim_hidden,
            num_components=num_components,
            depth=depth,
            negative_slope=negative_slope,
            beta=beta,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            num_examples=len(train_dataset),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=50,
            num_workers=0,
            seed=ensemble_id,
        )
        _ = outcome_model.fit(train_dataset=train_dataset, tune_dataset=valid_dataset)
    return -1


def tune_step(config):
    num_bins = config.get("num_bins")
    dim_hidden = config.get("dim_hidden")
    num_components = config.get("num_components")
    depth = config.get("depth")
    negative_slope = config.get("negative_slope")
    beta = config.get("beta")
    layer_norm = config.get("layer_norm")
    dropout_rate = config.get("dropout_rate")
    spectral_norm = config.get("spectral_norm")
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")

    dataset_name = config.get("dataset_name")
    config["ds_train"]["t_bins"] = num_bins
    config["ds_valid"]["t_bins"] = num_bins
    train_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    valid_dataset = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

    outcome_model = models.DiscreteTreatmentNeuralNetwork(
        job_dir=None,
        architecture="resnet",
        dim_input=train_dataset.dim_input,
        dim_treatment=train_dataset.dim_treatments,
        dim_output=train_dataset.dim_targets,
        dim_hidden=dim_hidden,
        num_components=num_components,
        depth=depth,
        negative_slope=negative_slope,
        beta=beta,
        layer_norm=layer_norm,
        spectral_norm=spectral_norm,
        dropout_rate=dropout_rate,
        num_examples=len(train_dataset),
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=epochs,
        num_workers=0,
        seed=config.get("seed"),
    )
    _ = outcome_model.fit(train_dataset, valid_dataset)


def tune_func(config):
    space = {
        "num_bins": tune.qlograndint(2, 32, 2),
        "dim_hidden": tune.qlograndint(256, 1024, 256),
        "depth": tune.randint(3, 6),
        "num_components": tune.randint(1, 32),
        "negative_slope": tune.quniform(0.0, 0.5, 0.01),
        "beta": tune.choice([0.0]),
        "layer_norm": tune.choice([False, True]),
        "dropout_rate": tune.quniform(0.0, 0.5, 0.01),
        "spectral_norm": tune.choice([0.0, 1.0, 3.0]),
        "learning_rate": tune.quniform(1e-4, 1e-3, 1e-4),
        "batch_size": tune.qlograndint(1024, 2048, 1024),
    }
    algorithm = bohb.TuneBOHB(space, metric="mean_loss", mode="min",)
    scheduler = schedulers.HyperBandForBOHB(
        time_attr="training_iteration", max_t=config.get("epochs"),
    )
    analysis = tune.run(
        run_or_experiment=tune_step,
        metric="mean_loss",
        mode="min",
        name="bohb",
        resources_per_trial={"gpu": config.get("gpu_per_model"),},
        num_samples=config.get("max_samples"),
        search_alg=algorithm,
        scheduler=scheduler,
        local_dir=config.get("experiment_dir"),
        config=config,
    )
    print("Best hyperparameters found are: ", analysis.best_config)
