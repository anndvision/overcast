import torch
from torch import nn
from torch import optim
from torch.utils import data

from ignite import metrics

from overcast import modules
from overcast.models import core
from overcast.models import utils
from overcast.metrics.regression import NegR2Score


class _TreatmentEffectNeuralNetwork(core.AuxiliaryTaskPyTorchModel):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_output,
        dim_hidden,
        depth,
        negative_slope,
        beta,
        layer_norm,
        spectral_norm,
        dropout_rate,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(_TreatmentEffectNeuralNetwork, self).__init__(
            job_dir, learning_rate, batch_size, epochs, num_workers, seed
        )

        self.feature_extractor = (
            nn.Sequential(
                modules.DenseLinear(
                    dim_input=dim_input,
                    dim_output=dim_hidden,
                    layer_norm=layer_norm,
                    spectral_norm=spectral_norm,
                ),
                modules.DenseFeatureExtractor(
                    architecture=architecture,
                    dim_input=dim_hidden,
                    dim_hidden=dim_hidden,
                    depth=depth - 1,
                    negative_slope=negative_slope,
                    layer_norm=layer_norm,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm,
                    activate_output=True,
                ),
            ).to(self.device)
            if depth > 0
            else nn.Identity()
        )

        self.beta = beta
        self.epsilon = 1.0

        self.batch_size = batch_size
        self.best_loss = 1e7
        self.patience = patience

        self.metrics = {
            "loss": NegR2Score(
                dim_output=dim_output,
                output_transform=lambda x: (
                    x["primary"]["outputs"].mean,
                    x["primary"]["targets"],
                ),
                device=self.device,
            ),
            "neg_log_prob": metrics.Average(
                output_transform=lambda x: -x["primary"]["outputs"]
                .log_prob(x["primary"]["targets"])
                .mean(),
                device=self.device,
            ),
            "neg_log_prob_task": metrics.Average(
                output_transform=lambda x: -x["auxiliary"]["outputs"]
                .log_prob(x["auxiliary"]["targets"])
                .mean(),
                device=self.device,
            ),
            "regularizer": metrics.Average(
                output_transform=lambda x: x["regularizer"], device=self.device,
            ),
        }

    def preprocess(self, batch):
        batch = (
            [x.to(self.device) for x in batch]
            if isinstance(batch, list)
            else batch.to(self.device)
        )
        return batch

    def train_step_primary(self, inputs, treatments, targets):
        self.network.train()
        self.optimizer.zero_grad()
        outputs = self.network(inputs=inputs, treatments=treatments)
        loss = -outputs.log_prob(targets).mean()
        loss.backward()
        self.optimizer.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step_primary(self, inputs, treatments, targets):
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(inputs=inputs, treatments=treatments)
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def train_step_auxiliary(self, inputs, targets):
        self.network_aux.train()
        self.optimizer_aux.zero_grad()
        outputs = self.network_aux(inputs)
        loss = -outputs.log_prob(targets).mean()
        loss.backward()
        self.optimizer_aux.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step_auxiliary(self, inputs, targets):
        self.network_aux.eval()
        with torch.no_grad():
            outputs = self.network_aux(inputs)
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def train_step_regularizer(self, inputs, treatments, targets):

        if self.beta > 0:
            self.network_aux.eval()
            self.network.train()

            with torch.no_grad():
                pi = self.network_aux(inputs)
                pi_t = torch.exp(pi.log_prob(treatments)).unsqueeze(-1) + 1e-7

            self.optimizer_re.zero_grad()
            outputs = self.network(inputs=inputs, treatments=treatments)

            psuedo_targets = targets - outputs.mean - self.epsilon * (treatments / pi_t)

            loss = self.beta * torch.square(psuedo_targets).mean()
            loss.backward()

            self.optimizer_dr.step()
        else:
            psuedo_targets = torch.zeros_like(targets)
            loss = torch.square(psuedo_targets).mean()
        return loss

    def tune_step_regularizer(self, inputs, treatments, targets):
        if self.beta > 0:
            self.network_aux.eval()
            self.network.eval()

            with torch.no_grad():
                pi = self.network_aux(inputs)
                pi_t = torch.exp(pi.log_prob(treatments)).unsqueeze(-1) + 1e-7
                outputs = self.network(inputs=inputs, treatments=treatments)

            psuedo_targets = targets - outputs.mean - self.epsilon * (treatments / pi_t)
            loss = self.beta * torch.square(psuedo_targets).mean()
        else:
            outputs = torch.zeros_like(targets)
            psuedo_targets = torch.zeros_like(targets)
            loss = torch.square(psuedo_targets).mean()
        return loss

    def train_step(self, engine, batch):
        inputs, treatments, targets = self.preprocess(batch)
        metrics_primary = self.train_step_primary(inputs, treatments, targets)
        metrics_auxiliary = self.train_step_auxiliary(inputs, treatments)
        metrics_regularizer = self.train_step_regularizer(inputs, treatments, targets)
        metric_values = {
            "primary": metrics_primary,
            "auxiliary": metrics_auxiliary,
            "regularizer": metrics_regularizer,
        }
        return metric_values

    def tune_step(self, engine, batch):
        inputs, treatments, targets = self.preprocess(batch)
        metrics_primary = self.tune_step_primary(inputs, treatments, targets)
        metrics_auxiliary = self.tune_step_auxiliary(inputs, treatments)
        metrics_regularizer = self.tune_step_regularizer(inputs, treatments, targets)
        metric_values = {
            "primary": metrics_primary,
            "auxiliary": metrics_auxiliary,
            "regularizer": metrics_regularizer,
        }
        return metric_values

    def preprocess_treatment(self, treatment, treatments, xfm=None):
        raise NotImplementedError()

    def predict_mean(self, dataset, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mean = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, treatments, _ = self.preprocess(batch)
                mean.append(self.network(inputs=inputs, treatments=treatments).mean)
        if dataset.targets_xfm is not None:
            mean = dataset.targets_xfm.inverse_transform(
                torch.cat(mean, dim=0).to("cpu").numpy()
            )
        return mean

    def sample(self, dataset, num_samples, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        y = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, treatments, _ = self.preprocess(batch)
                y.append(
                    self.network(inputs=inputs, treatments=treatments).sample(
                        torch.Size([num_samples])
                    )
                )
        if dataset.targets_xfm is not None:
            y = dataset.targets_xfm.inverse_transform(
                torch.cat(y, dim=0).to("cpu").numpy()
            )
        return y

    def sample_po(self, dataset, treatment, num_samples):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        y = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, treatments, _ = self.preprocess(batch)
                y.append(
                    self.network(inputs=inputs, treatments=treatments).sample(
                        torch.Size([num_samples])
                    )
                )
        if dataset.targets_xfm is not None:
            y = dataset.targets_xfm.inverse_transform(
                torch.cat(y, dim=0).to("cpu").numpy()
            )
        return y

    def predict_aux_mean(self, dataset, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mean = []
        self.network_aux.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, _, _ = self.preprocess(batch)
                mean.append(self.network_aux(inputs).mean)
        if dataset.treatments_xfm is not None:
            mean = dataset.treatments_xfm.inverse_transform(
                torch.cat(mean, dim=0).to("cpu").numpy()
            )
        return mean

    def sample_aux(self, dataset, num_samples, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        t = []
        self.network_aux.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, _, _ = self.preprocess(batch)
                t.append(self.network_aux(inputs).sample(torch.Size([num_samples])))
        if dataset.treatments_xfm is not None:
            t = dataset.treatments_xfm.inverse_transform(
                torch.cat(t, dim=0).to("cpu").numpy()
            )
        return t

    def predict_capo(self, dataset, treatment, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        capo = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, treatments, _ = self.preprocess(batch)
                t = self.preprocess_treatment(
                    treatment=treatment,
                    treatments=treatments,
                    xfm=dataset.treatments_xfm,
                )
                capo.append(self.network(inputs=inputs, treatments=t).mean)
        capo = torch.cat(capo, dim=0).to("cpu").numpy()
        if dataset.targets_xfm is not None:
            capo = dataset.targets_xfm.inverse_transform(capo)
        return capo

    def sample_po(self, dataset, treatment, num_samples=100, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        y = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                inputs, treatments, _ = self.preprocess(batch)
                t = self.preprocess_treatment(
                    treatment=treatment,
                    treatments=treatments,
                    xfm=dataset.treatments_xfm,
                )
                y.append(
                    self.network(inputs=inputs, treatments=t).sample(
                        torch.Size([num_samples])
                    )
                )
        if dataset.targets_xfm is not None:
            y = dataset.targets_xfm.inverse_transform(
                torch.cat(y, dim=1).to("cpu").numpy()
            )
        return y


class DiscreteTreatmentNeuralNetwork(_TreatmentEffectNeuralNetwork):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_treatment,
        dim_output,
        num_components,
        dim_hidden,
        depth,
        negative_slope,
        beta,
        layer_norm,
        spectral_norm,
        dropout_rate,
        num_examples,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(DiscreteTreatmentNeuralNetwork, self).__init__(
            job_dir=job_dir,
            architecture=architecture,
            dim_input=dim_input,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            depth=depth,
            negative_slope=negative_slope,
            beta=beta,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
        )

        self.network_aux = modules.NeuralDensityNetwork(
            feature_extractor=self.feature_extractor,
            density_estimator=modules.Categorical(
                dim_input=dim_hidden, dim_output=dim_treatment,
            ),
        )
        self.network = modules.TarNet(
            feature_extractor=self.feature_extractor,
            hypotheses=modules.GroupIdentity(),
            density_estimator=modules.GroupGMM(
                num_components=num_components,
                dim_input=dim_hidden,
                dim_output=dim_output,
                groups=dim_treatment,
            ),
        )
        self.optimizer_aux = optim.Adam(
            params=self.network_aux.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.optimizer_regularizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.network_aux.to(self.device)
        self.network.to(self.device)

    def preprocess_treatment(self, treatment, treatments, xfm=None):
        t = torch.zeros_like(treatments)
        t[:, treatment] = 1
        return t

    def predict_capo_interval(
        self, dataset, treatment, log_lambda, num_samples=100, batch_size=None,
    ):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        _lambda = torch.exp(torch.tensor([log_lambda])).to(self.device) + utils.eps
        # predict and sample
        upper, lower = [], []
        self.network.eval()
        self.network_aux.eval()
        for batch in dl:
            with torch.no_grad():
                inputs, treatments, _ = self.preprocess(batch)
                t = self.preprocess_treatment(
                    treatment=treatment,
                    treatments=treatments,
                    xfm=dataset.treatments_xfm,
                )
                y_density = self.network(inputs=inputs, treatments=t)
                y_samples = y_density.sample(
                    torch.Size([num_samples])
                )  # [num_samples, batch_size, dy]
                mu = y_density.mean.unsqueeze(0)  # [1, batch_size, dy]
                t_density = self.network_aux(inputs)
                pi = (
                    torch.exp(t_density.log_prob(t)).unsqueeze(0).unsqueeze(-1)
                )  # [1, batch_size, 1]
                pi = torch.clip(pi, utils.eps, 1 - utils.eps)
                # get alpha prime
                alpha = utils.alpha_func(pi, _lambda)
                beta = utils.beta_func(pi, _lambda)
                alpha_prime = alpha / (beta - alpha)
                # sweep over upper bounds
                r = y_samples - mu  # [num_samples, batch_size, dy]
                d = y_samples - y_samples.unsqueeze(
                    1
                )  # [num_samples, num_samples, batch_size, dy]
                h_u = torch.heaviside(
                    d, torch.tensor([1.0], device=self.device)
                )  # [num_samples, num_samples, batch_size, dy]
                numer_upper = (h_u * r.unsqueeze(0)).mean(
                    1
                )  # [num_samples, batch_size, dy]
                denom_upper = (
                    h_u.mean(1) + alpha_prime + utils.eps
                )  # [num_samples, batch_size, dy]
                upper_batch = (
                    mu + numer_upper / denom_upper
                )  # [num_samples, batch_size, dy]
                upper_batch = upper_batch.max(0)[0]  # [batch_size, dy]
                upper.append(upper_batch)
                # sweep over lower bounds
                h_l = torch.heaviside(
                    -d, torch.tensor([1.0], device=self.device)
                )  # [num_samples, num_samples, batch_size, dy]
                numer_lower = (h_l * r.unsqueeze(0)).mean(
                    1
                )  # [num_samples, batch_size, dy]
                denom_lower = (
                    h_l.mean(1) + alpha_prime + utils.eps
                )  # [num_samples, batch_size, dy]
                lower_batch = (
                    mu + numer_lower / denom_lower
                )  # [num_samples, batch_size, dy]
                lower_batch = lower_batch.min(0)[0]  # [batch_size, dy]
                lower.append(lower_batch)
        # post process
        upper = torch.cat(upper, dim=0).to("cpu").numpy()
        lower = torch.cat(lower, dim=0).to("cpu").numpy()
        if dataset.targets_xfm is not None:
            upper = dataset.targets_xfm.inverse_transform(upper)
            lower = dataset.targets_xfm.inverse_transform(lower)
        return lower, upper


class _ContinousTreatmentNeuralNetwork(_TreatmentEffectNeuralNetwork):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_treatment,
        dim_output,
        num_components_treatment,
        dim_hidden,
        depth,
        negative_slope,
        beta,
        layer_norm,
        spectral_norm,
        dropout_rate,
        num_examples,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(_ContinousTreatmentNeuralNetwork, self).__init__(
            job_dir=job_dir,
            architecture=architecture,
            dim_input=dim_input,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            depth=depth,
            negative_slope=negative_slope,
            beta=beta,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
        )
        self.network_aux = modules.NeuralDensityNetwork(
            feature_extractor=self.feature_extractor,
            density_estimator=modules.GMM(
                num_components=num_components_treatment,
                dim_input=dim_hidden,
                dim_output=dim_treatment,
            ),
        )
        self.optimizer_aux = optim.Adam(
            params=self.network_aux.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.network_aux.to(self.device)

    def preprocess_treatment(self, treatment, treatments, xfm=None):
        treatment = (
            xfm.transform([[treatment]]).item() if xfm is not None else treatment
        )
        return treatment * torch.ones_like(treatments)


class AppendedTreatmentNeuralNetwork(_ContinousTreatmentNeuralNetwork):
    def __init__(
        self,
        job_dir,
        architecture,
        dim_input,
        dim_treatment,
        dim_output,
        num_components_outcome,
        num_components_treatment,
        dim_hidden,
        depth,
        negative_slope,
        beta,
        layer_norm,
        spectral_norm,
        dropout_rate,
        num_examples,
        learning_rate,
        batch_size,
        epochs,
        patience,
        num_workers,
        seed,
    ):
        super(AppendedTreatmentNeuralNetwork, self).__init__(
            job_dir=job_dir,
            architecture=architecture,
            dim_input=dim_input,
            dim_output=dim_output,
            dim_treatment=dim_treatment,
            num_components_treatment=num_components_treatment,
            dim_hidden=dim_hidden,
            depth=depth,
            negative_slope=negative_slope,
            beta=beta,
            layer_norm=layer_norm,
            spectral_norm=spectral_norm,
            dropout_rate=dropout_rate,
            num_examples=num_examples,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
        )
        self.network = modules.AppendedDensityNetwork(
            feature_extractor=self.feature_extractor,
            density_estimator=nn.Sequential(
                modules.DenseLinear(
                    dim_input=dim_hidden + 1,
                    dim_output=dim_hidden * 2,
                    layer_norm=layer_norm,
                    spectral_norm=spectral_norm,
                ),
                modules.DenseFeatureExtractor(
                    architecture=architecture,
                    dim_input=dim_hidden * 2,
                    dim_hidden=dim_hidden * 2,
                    depth=2,
                    negative_slope=negative_slope,
                    layer_norm=layer_norm,
                    dropout_rate=dropout_rate,
                    spectral_norm=spectral_norm,
                    activate_output=True,
                ),
                modules.GMM(
                    num_components=num_components_outcome,
                    dim_input=dim_hidden * 2,
                    dim_output=dim_output,
                ),
            ),
        )
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.optimizer_regularizer = optim.Adam(
            params=self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=(0.5 * (1 - dropout_rate)) / num_examples,
        )
        self.network.to(self.device)

    def predict_capo_interval(
        self, dataset, treatment, log_lambda, num_samples=100, batch_size=None,
    ):
        dl = data.DataLoader(
            dataset,
            batch_size=2 * self.batch_size if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        _lambda = torch.exp(torch.tensor([log_lambda])).to(self.device) + utils.eps
        alpha_prime = 1 / (_lambda ** 2 - 1)
        # predict and sample
        upper, lower = [], []
        self.network.eval()
        self.network_aux.eval()
        for batch in dl:
            with torch.no_grad():
                inputs, treatments, _ = self.preprocess(batch)
                t = self.preprocess_treatment(
                    treatment=treatment,
                    treatments=treatments,
                    xfm=dataset.treatments_xfm,
                )
                y_density = self.network(inputs=inputs, treatments=t)
                y_samples = y_density.sample(
                    torch.Size([num_samples])
                )  # [num_samples, batch_size, dy]
                mu = y_density.mean.unsqueeze(0)  # [1, batch_size, dy]
                t_density = self.network_aux(inputs)
                # sweep over upper bounds
                r = y_samples - mu  # [num_samples, batch_size, dy]
                d = y_samples - y_samples.unsqueeze(
                    1
                )  # [num_samples, num_samples, batch_size, dy]
                h_u = torch.heaviside(
                    d, torch.tensor([1.0], device=self.device)
                )  # [num_samples, num_samples, batch_size, dy]
                numer_upper = (h_u * r.unsqueeze(0)).mean(
                    1
                )  # [num_samples, batch_size, dy]
                denom_upper = (
                    h_u.mean(1) + alpha_prime + utils.eps
                )  # [num_samples, batch_size, dy]
                upper_batch = (
                    mu + numer_upper / denom_upper
                )  # [num_samples, batch_size, dy]
                upper_batch = upper_batch.max(0)[0]  # [batch_size, dy]
                upper.append(upper_batch)
                # sweep over lower bounds
                h_l = torch.heaviside(
                    -d, torch.tensor([1.0], device=self.device)
                )  # [num_samples, num_samples, batch_size, dy]
                numer_lower = (h_l * r.unsqueeze(0)).mean(
                    1
                )  # [num_samples, batch_size, dy]
                denom_lower = (
                    h_l.mean(1) + alpha_prime + utils.eps
                )  # [num_samples, batch_size, dy]
                lower_batch = (
                    mu + numer_lower / denom_lower
                )  # [num_samples, batch_size, dy]
                lower_batch = lower_batch.min(0)[0]  # [batch_size, dy]
                lower.append(lower_batch)
        # post process
        upper = torch.cat(upper, dim=0).to("cpu").numpy()
        lower = torch.cat(lower, dim=0).to("cpu").numpy()
        if dataset.targets_xfm is not None:
            upper = dataset.targets_xfm.inverse_transform(upper)
            lower = dataset.targets_xfm.inverse_transform(lower)
        return lower, upper
