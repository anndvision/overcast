import os
import copy
import torch

from abc import ABC

from ray import tune

from ignite import utils
from ignite import engine
from ignite import handlers
from ignite import distributed
from ignite.contrib.handlers import tensorboard_logger

from torch.utils import data


class BaseModel(ABC):
    def __init__(self, job_dir, seed):
        super(BaseModel, self).__init__()
        self.job_dir = job_dir
        self.seed = seed

    def fit(self, train_dataset, tune_dataset):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement train()"
        )

    def save(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement save()"
        )

    def load(self):
        raise NotImplementedError(
            "Classes that inherit from BaseModel must implement load()"
        )


class PyTorchModel(BaseModel):
    def __init__(self, job_dir, learning_rate, batch_size, epochs, num_workers, seed):
        super(PyTorchModel, self).__init__(job_dir=job_dir, seed=seed)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.logger = utils.setup_logger(
            name=__name__ + "." + self.__class__.__name__, distributed_rank=0
        )
        self.trainer = engine.Engine(self.train_step)
        self.evaluator = engine.Engine(self.tune_step)
        self.timer = handlers.Timer(average=True)
        self._network = None
        self._optimizer = None
        self._metrics = None
        self.num_workers = num_workers
        self.device = distributed.device()
        self.best_state = {}
        self.counter = 0

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        self._metrics = value

    def train_step(self, trainer, batch):
        raise NotImplementedError()

    def tune_step(self, trainer, batch):
        raise NotImplementedError()

    def preprocess(self, batch):
        raise NotImplementedError()

    def fit(self, train_dataset, tune_dataset):
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        tune_loader = data.DataLoader(
            tune_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
        )
        # Instantiate trainer
        for k, v in self.metrics.items():
            v.attach(self.trainer, k)
            v.attach(self.evaluator, k)
        self.trainer.add_event_handler(
            engine.Events.EPOCH_COMPLETED,
            self.on_epoch_completed,
            train_loader,
            tune_loader,
        )
        self.trainer.add_event_handler(
            engine.Events.COMPLETED, self.on_training_completed, tune_loader
        )
        self.timer.attach(
            self.trainer,
            start=engine.Events.STARTED,
            resume=engine.Events.EPOCH_STARTED,
            pause=engine.Events.EPOCH_COMPLETED,
            step=engine.Events.EPOCH_COMPLETED,
        )
        if not tune.is_session_enabled():
            tb_logger = tensorboard_logger.TensorboardLogger(log_dir=self.job_dir)
            tb_logger.attach_output_handler(
                self.evaluator,
                event_name=engine.Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=list(self.metrics.keys()),  # TODO: generalize metric names
                global_step_transform=handlers.global_step_from_engine(self.trainer),
            )
        # Train
        self.trainer.run(train_loader, max_epochs=self.epochs)
        return self.evaluator.state.metrics

    def on_epoch_completed(self, engine, train_loader, tune_loader):
        train_metrics = self.trainer.state.metrics
        print("Metrics Epoch", engine.state.epoch)
        justify = max(len(k) for k in train_metrics) + 2
        time_str = "time (seconds)"
        print(f"{time_str:<{justify}} {self.timer.value():<5f}")
        for k, v in train_metrics.items():
            if type(v) == float:
                print("train {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
            print("train {:<{justify}} {:<5}".format(k, v, justify=justify))
        self.evaluator.run(tune_loader)
        tune_metrics = self.evaluator.state.metrics
        if tune.is_session_enabled():
            tune.report(
                training_iteration=engine.state.epoch, mean_loss=tune_metrics["loss"]
            )
        justify = max(len(k) for k in tune_metrics) + 2
        for k, v in tune_metrics.items():
            if type(v) == float:
                print("tune {:<{justify}} {:<5f}".format(k, v, justify=justify))
                continue
        if tune_metrics["loss"] < self.best_loss:
            self.best_loss = tune_metrics["loss"]
            self.counter = 0
            self.update()
        else:
            self.counter += 1
        if self.counter == self.patience:
            self.logger.info(
                "Early Stopping: No improvement for {} epochs".format(self.patience)
            )
            engine.terminate()

    def on_training_completed(self, engine, loader):
        self.save()
        self.load()
        if not tune.is_session_enabled():
            self.evaluator.run(loader)
            metric_values = self.evaluator.state.metrics
            print("Metrics Epoch", engine.state.epoch)
            justify = max(len(k) for k in metric_values) + 2
            for k, v in metric_values.items():
                if type(v) == float:
                    print("best {:<{justify}} {:<5f}".format(k, v, justify=justify))
                    continue

    def update(self):
        if not tune.is_session_enabled():
            self.best_state.update(
                {
                    "network": copy.deepcopy(self.network.state_dict()),
                    "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                    "engine": copy.copy(self.trainer.state),
                }
            )

    def save(self):
        if not tune.is_session_enabled():
            p = os.path.join(self.job_dir, "best_checkpoint.pt")
            torch.save(self.best_state, p)

    def load(self):
        if tune.is_session_enabled():
            with tune.checkpoint_dir(step=self.trainer.state.epoch) as checkpoint_dir:
                p = os.path.join(checkpoint_dir, "checkpoint.pt")
        else:
            file_name = "best_checkpoint.pt"
            p = os.path.join(self.job_dir, file_name)
        if not os.path.exists(p):
            self.logger.info(
                "Checkpoint {} does not exist, starting a new engine".format(p)
            )
            return
        self.logger.info("Loading saved checkpoint {}".format(p))
        checkpoint = torch.load(p)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.trainer.state = checkpoint["engine"]

    def predict_mean(self, dataset):
        raise NotImplementedError()

    def sample(self, dataset, num_samples):
        raise NotImplementedError()


class AuxiliaryTaskPyTorchModel(PyTorchModel):
    def __init__(self, job_dir, learning_rate, batch_size, epochs, num_workers, seed):
        super(AuxiliaryTaskPyTorchModel, self).__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=num_workers,
            seed=seed,
        )
        self._network_aux = None
        self._optimizer_aux = None

    @property
    def network_aux(self):
        return self._network_aux

    @network_aux.setter
    def network_aux(self, value):
        self._network_aux = value

    @property
    def optimizer_aux(self):
        return self._optimizer_aux

    @optimizer_aux.setter
    def optimizer_aux(self, value):
        self._optimizer_aux = value

    def update(self):
        super(AuxiliaryTaskPyTorchModel, self).update()
        if not tune.is_session_enabled():
            self.best_state.update(
                {
                    "network_aux": copy.deepcopy(self.network_aux.state_dict()),
                    "optimizer_aux": copy.deepcopy(self.optimizer_aux.state_dict()),
                }
            )

    def load(self):
        super(AuxiliaryTaskPyTorchModel, self).load()
        if tune.is_session_enabled():
            with tune.checkpoint_dir(step=self.trainer.state.epoch) as checkpoint_dir:
                p = os.path.join(checkpoint_dir, "checkpoint.pt")
        else:
            file_name = "best_checkpoint.pt"
            p = os.path.join(self.job_dir, file_name)
        if not os.path.exists(p):
            self.logger.info(
                "Checkpoint {} does not exist, starting a new engine".format(p)
            )
            return
        self.logger.info("Loading saved checkpoint {}".format(p))
        checkpoint = torch.load(p)
        self.network_aux.load_state_dict(checkpoint["network_aux"])
        self.optimizer_aux.load_state_dict(checkpoint["optimizer_aux"])

    def predict_aux_mean(self, dataset):
        raise NotImplementedError()

    def sample_aux(self, dataset, num_samples):
        raise NotImplementedError()
