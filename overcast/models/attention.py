import torch
from torch import nn
from torch import optim
from torch.utils import data

from ignite import metrics

from overcast.models import core
from overcast.models import utils
from overcast import modules
from overcast.metrics.regression import NegR2Score


class _TreatmentEffectAttentionNetwork(core.AuxiliaryTaskPyTorchModel):
    def __init__(
        self,
        job_dir,
        dim_input,
        dim_output,
        dim_hidden,
        depth,
        num_heads,
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
        super(_TreatmentEffectAttentionNetwork, self).__init__(
            job_dir, learning_rate, batch_size, epochs, num_workers, seed
        )

        self.feature_extractor = modules.Encoder(
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            depth=depth,
            num_heads=num_heads,
            layer_norm=layer_norm,
            negative_slope=negative_slope,
            dropout_rate=dropout_rate,
            spectral_norm=spectral_norm,
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
        }

    def preprocess(self, batch):
        batch = (
            [x.to(self.device) for x in batch]
            if isinstance(batch, list)
            else batch.to(self.device)
        )
        inputs, treatments, targets, position = batch
        inputs_mask = inputs[:, :, :1].isnan().transpose(-2, -1) == False
        outputs_mask = targets[:, :, 0].isnan().reshape(-1) == False
        outputs_mask = outputs_mask & inputs_mask.reshape(-1)
        inputs[inputs.isnan()] = 0.0
        treatments[treatments.isnan()] = 0.0
        targets[targets.isnan()] = 0.0
        position[position.isnan()] = 0.0
        return inputs, treatments, targets, position, inputs_mask, outputs_mask

    def train_step_primary(
        self, inputs, treatments, targets, position, inputs_mask, outputs_mask,
    ):
        self.network.train()
        self.optimizer.zero_grad()
        outputs = self.network(
            inputs=inputs,
            treatments=treatments,
            position=position,
            inputs_mask=inputs_mask,
            outputs_mask=outputs_mask,
        )
        targets = targets.reshape(-1, targets.shape[-1])[outputs_mask]
        loss = -outputs.log_prob(targets).mean()
        loss.backward()
        self.optimizer.step()
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def tune_step_primary(
        self, inputs, treatments, targets, position, inputs_mask, outputs_mask,
    ):
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(
                inputs=inputs,
                treatments=treatments,
                position=position,
                inputs_mask=inputs_mask,
                outputs_mask=outputs_mask,
            )
        targets = targets.reshape(-1, targets.shape[-1])[outputs_mask]
        metric_values = {
            "outputs": outputs,
            "targets": targets,
        }
        return metric_values

    def train_step_auxiliary(
        self, inputs, treatments, position, inputs_mask, outputs_mask,
    ):
        self.network_aux.train()
        self.optimizer_aux.zero_grad()
        outputs = self.network_aux(
            inputs=inputs,
            outputs=treatments,
            position=position,
            inputs_mask=inputs_mask,
            outputs_mask=outputs_mask,
        )
        treatments = treatments.reshape(-1, treatments.shape[-1])[outputs_mask]
        loss = -outputs.log_prob(treatments).mean()
        loss.backward()
        self.optimizer_aux.step()
        metric_values = {
            "outputs": outputs,
            "targets": treatments,
        }
        return metric_values

    def tune_step_auxiliary(
        self, inputs, treatments, position, inputs_mask, outputs_mask,
    ):
        self.network_aux.eval()
        with torch.no_grad():
            outputs = self.network_aux(
                inputs=inputs,
                outputs=treatments,
                position=position,
                inputs_mask=inputs_mask,
                outputs_mask=outputs_mask,
            )
        treatments = treatments.reshape(-1, treatments.shape[-1])[outputs_mask]
        metric_values = {
            "outputs": outputs,
            "targets": treatments,
        }
        return metric_values

    def train_step(self, engine, batch):
        (
            inputs,
            treatments,
            targets,
            position,
            inputs_mask,
            outputs_mask,
        ) = self.preprocess(batch)
        metrics_primary = self.train_step_primary(
            inputs=inputs,
            treatments=treatments,
            targets=targets,
            position=position,
            inputs_mask=inputs_mask,
            outputs_mask=outputs_mask,
        )
        metrics_auxiliary = self.train_step_auxiliary(
            inputs=inputs,
            treatments=treatments,
            position=position,
            inputs_mask=inputs_mask,
            outputs_mask=outputs_mask,
        )
        metric_values = {
            "primary": metrics_primary,
            "auxiliary": metrics_auxiliary,
        }
        return metric_values

    def tune_step(self, engine, batch):
        (
            inputs,
            treatments,
            targets,
            position,
            inputs_mask,
            outputs_mask,
        ) = self.preprocess(batch)
        metrics_primary = self.tune_step_primary(
            inputs=inputs,
            treatments=treatments,
            targets=targets,
            position=position,
            inputs_mask=inputs_mask,
            outputs_mask=outputs_mask,
        )
        metrics_auxiliary = self.tune_step_auxiliary(
            inputs=inputs,
            treatments=treatments,
            position=position,
            inputs_mask=inputs_mask,
            outputs_mask=outputs_mask,
        )
        metric_values = {
            "primary": metrics_primary,
            "auxiliary": metrics_auxiliary,
        }
        return metric_values

    def preprocess_treatment(self, treatment, treatments, xfm=None):
        raise NotImplementedError()

    def predict_mean(self, dataset, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mean = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                (
                    inputs,
                    treatments,
                    _,
                    position,
                    inputs_mask,
                    outputs_mask,
                ) = self.preprocess(batch)
                y_density = self.network(
                    inputs=inputs,
                    treatments=treatments,
                    position=position,
                    inputs_mask=inputs_mask,
                    outputs_mask=outputs_mask,
                )
                mean.append(y_density.mean)
        if dataset.targets_xfm is not None:
            mean = dataset.targets_xfm.inverse_transform(
                torch.cat(mean, dim=0).to("cpu").numpy()
            )
        return mean

    def sample(self, dataset, num_samples):
        dl = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        y = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                (
                    inputs,
                    treatments,
                    _,
                    position,
                    inputs_mask,
                    outputs_mask,
                ) = self.preprocess(batch)
                y_density = self.network(
                    inputs=inputs,
                    treatments=treatments,
                    position=position,
                    inputs_mask=inputs_mask,
                    outputs_mask=outputs_mask,
                )
                y.append(y_density.sample(torch.Size([num_samples])))
        if dataset.targets_xfm is not None:
            y = dataset.targets_xfm.inverse_transform(
                torch.cat(y, dim=0).to("cpu").numpy()
            )
        return y

    def predict_aux_mean(self, dataset, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        mean = []
        self.network_aux.eval()
        with torch.no_grad():
            for batch in dl:
                (
                    inputs,
                    treatments,
                    _,
                    position,
                    inputs_mask,
                    outputs_mask,
                ) = self.preprocess(batch)
                aux_density = self.network_aux(
                    inputs=inputs,
                    outputs=treatments,
                    position=position,
                    inputs_mask=inputs_mask,
                    outputs_mask=outputs_mask,
                )
                mean.append(aux_density.mean)
        if dataset.treatments_xfm is not None:
            mean = dataset.treatments_xfm.inverse_transform(
                torch.cat(mean, dim=0).to("cpu").numpy()
            )
        return mean

    def sample_aux(self, dataset, num_samples):
        dl = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        t = []
        self.network_aux.eval()
        with torch.no_grad():
            for batch in dl:
                (
                    inputs,
                    treatments,
                    _,
                    position,
                    inputs_mask,
                    outputs_mask,
                ) = self.preprocess(batch)
                aux_density = self.network_aux(
                    inputs=inputs,
                    outputs=treatments,
                    position=position,
                    inputs_mask=inputs_mask,
                    outputs_mask=outputs_mask,
                )
                t.append(aux_density.sample(torch.Size([num_samples])))
        if dataset.treatments_xfm is not None:
            t = dataset.treatments_xfm.inverse_transform(
                torch.cat(t, dim=0).to("cpu").numpy()
            )
        return t

    def predict_capo(self, dataset, treatment, batch_size=None):
        dl = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        capo = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                (
                    inputs,
                    treatments,
                    _,
                    position,
                    inputs_mask,
                    outputs_mask,
                ) = self.preprocess(batch)
                t = self.preprocess_treatment(
                    treatment=treatment,
                    treatments=treatments,
                    xfm=dataset.treatments_xfm,
                )
                y_density = self.network(
                    inputs=inputs,
                    treatments=t,
                    position=position,
                    inputs_mask=inputs_mask,
                    outputs_mask=outputs_mask,
                )
                capo.append(y_density.mean)
        capo = torch.cat(capo, dim=0).to("cpu").numpy()
        if dataset.targets_xfm is not None:
            capo = dataset.targets_xfm.inverse_transform(capo)
        return capo

    def sample_po(self, dataset, treatment, num_samples=100):
        dl = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        y = []
        self.network.eval()
        with torch.no_grad():
            for batch in dl:
                (
                    inputs,
                    treatments,
                    _,
                    position,
                    inputs_mask,
                    outputs_mask,
                ) = self.preprocess(batch)
                t = self.preprocess_treatment(
                    treatment=treatment,
                    treatments=treatments,
                    xfm=dataset.treatments_xfm,
                )
                y_density = self.network(
                    inputs=inputs,
                    treatments=t,
                    position=position,
                    inputs_mask=inputs_mask,
                    outputs_mask=outputs_mask,
                )
                y.append(y_density.sample(torch.Size([num_samples])))
        if dataset.targets_xfm is not None:
            y = dataset.targets_xfm.inverse_transform(
                torch.cat(y, dim=1).to("cpu").numpy()
            )
        return y


class DiscreteTreatmentAttentionNetwork(_TreatmentEffectAttentionNetwork):
    def __init__(
        self,
        job_dir,
        dim_input,
        dim_treatment,
        dim_output,
        num_components,
        dim_hidden,
        depth,
        num_heads,
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
        super(DiscreteTreatmentAttentionNetwork, self).__init__(
            job_dir=job_dir,
            dim_input=dim_input,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            depth=depth,
            num_heads=num_heads,
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

        self.network_aux = modules.DensityAttentionNetwork(
            feature_extractor=self.feature_extractor,
            density_estimator=modules.Categorical(
                dim_input=dim_hidden, dim_output=dim_treatment,
            ),
        )
        self.network = modules.TarAttentionNetwork(
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
        self.network_aux.to(self.device)
        self.network.to(self.device)

    def preprocess_treatment(self, treatment, treatments, xfm=None):
        t = torch.zeros_like(treatments)
        t[:, :, treatment] = 1
        return t

    def predict_capo_interval(
        self, dataset, treatment, log_lambda, num_samples=100, batch_size=None,
    ):
        dl = data.DataLoader(
            dataset,
            batch_size=1,
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
                (
                    inputs,
                    treatments,
                    _,
                    position,
                    inputs_mask,
                    outputs_mask,
                ) = self.preprocess(batch)
                t = self.preprocess_treatment(
                    treatment=treatment,
                    treatments=treatments,
                    xfm=dataset.treatments_xfm,
                )
                y_density = self.network(
                    inputs=inputs,
                    treatments=t,
                    position=position,
                    inputs_mask=inputs_mask,
                    outputs_mask=outputs_mask,
                )
                y_samples = y_density.sample(
                    torch.Size([num_samples])
                )  # [num_samples, batch_size, dy]
                mu = y_density.mean.unsqueeze(0)  # [1, batch_size, dy]
                pi = (
                    self.network_aux(
                        inputs=inputs,
                        outputs=treatments,
                        position=position,
                        inputs_mask=inputs_mask,
                        outputs_mask=outputs_mask,
                    )
                    .probs[:, treatment]
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
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


class _ContinousTreatmentAttentionNetwork(_TreatmentEffectAttentionNetwork):
    def __init__(
        self,
        job_dir,
        dim_input,
        dim_treatment,
        dim_output,
        num_components_treatment,
        dim_hidden,
        depth,
        num_heads,
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
        super(_ContinousTreatmentAttentionNetwork, self).__init__(
            job_dir=job_dir,
            dim_input=dim_input,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            depth=depth,
            num_heads=num_heads,
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
        self.network_aux = modules.DensityAttentionNetwork(
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

    def predict_capo_interval(
        self, dataset, treatment, log_lambda, num_samples=100, batch_size=None,
    ):
        dl = data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        _lambda = torch.exp(torch.tensor([log_lambda])).to(self.device) + utils.eps
        # predict and sample
        alpha_prime = 1 / (_lambda ** 2 - 1)
        upper, lower = [], []
        self.network.eval()
        self.network_aux.eval()
        for batch in dl:
            with torch.no_grad():
                (
                    inputs,
                    treatments,
                    _,
                    position,
                    inputs_mask,
                    outputs_mask,
                ) = self.preprocess(batch)
                t = self.preprocess_treatment(
                    treatment=treatment,
                    treatments=treatments,
                    xfm=dataset.treatments_xfm,
                )
                y_density = self.network(
                    inputs=inputs,
                    treatments=t,
                    position=position,
                    inputs_mask=inputs_mask,
                    outputs_mask=outputs_mask,
                )
                y_samples = y_density.sample(
                    torch.Size([num_samples])
                )  # [num_samples, batch_size, dy]
                mu = y_density.mean.unsqueeze(0)  # [1, batch_size, dy]
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


class AppendedTreatmentAttentionNetwork(_ContinousTreatmentAttentionNetwork):
    def __init__(
        self,
        job_dir,
        dim_input,
        dim_treatment,
        dim_output,
        num_components_outcome,
        num_components_treatment,
        dim_hidden,
        depth,
        num_heads,
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
        super(AppendedTreatmentAttentionNetwork, self).__init__(
            job_dir=job_dir,
            dim_input=dim_input,
            dim_output=dim_output,
            dim_treatment=dim_treatment,
            num_components_treatment=num_components_treatment,
            dim_hidden=dim_hidden,
            depth=depth,
            num_heads=num_heads,
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
        self.network = modules.AppendedDensityAttentionNetwork(
            feature_extractor=self.feature_extractor,
            density_estimator=nn.Sequential(
                modules.DenseLinear(
                    dim_input=dim_hidden + 1,
                    dim_output=dim_hidden * 2,
                    layer_norm=layer_norm,
                    spectral_norm=spectral_norm,
                ),
                modules.DenseFeatureExtractor(
                    architecture="resnet",
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
        self.network.to(self.device)
