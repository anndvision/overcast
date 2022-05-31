# overcast

| **[Abstract](#abstract)**
| **[Citation](#citation)**
| **[Installation](#installation)**
| **[Data](#data)**
| **[Quince Model](#quince)**

Code to reproduce the results presented in [Scalable Sensitivity and Uncertainty Analyses for Causal-Effect Estimates of Continuous-Valued Interventions](https://arxiv.org/abs/2204.10022).

## Abstract

Estimating the effects of continuous-valued interventions from observational data is a critically important task for climate science, healthcare, and economics. Recent work focuses on designing neural network architectures and regularization functions to allow for scalable estimation of average and individual-level dose-response curves from high-dimensional, large-sample data. Such methodologies assume ignorability (observation of all confounding variables) and positivity (observation of all treatment levels for every covariate value describing a set of units), assumptions problematic in the continuous treatment regime. Scalable sensitivity and uncertainty analyses to understand the ignorance induced in causal estimates when these assumptions are relaxed are less studied. Here, we develop a continuous treatment-effect marginal sensitivity model (CMSM) and derive bounds that agree with the observed data and a researcher-defined level of hidden confounding. We introduce a scalable algorithm and uncertainty-aware deep models to derive and estimate these bounds for high-dimensional, large-sample observational data. We work in concert with climate scientists interested in the climatological impacts of human emissions on cloud properties using satellite observations from the past 15 years. This problem is known to be complicated by many unobserved confounders.

## Citation

If you find this code useful for your own work, please cite:

```bibtex
@article{jesson2022scalable,
  title={Scalable Sensitivity and Uncertainty Analysis for Causal-Effect Estimates of Continuous-Valued Interventions},
  author={Jesson, Andrew and Douglas, Alyson and Manshausen, Peter and Meinshausen, Nicolai and Stier, Philip and Gal, Yarin and Shalit, Uri},
  journal={arXiv preprint arXiv:2204.10022},
  year={2022}
}
```

## Installation

```.sh
git clone git@github.com:anndvision/overcast.git
cd overcast
conda env create -f environment.yml
conda activate overcast
pip install -e .
```

## Data

Make a directory to store the data and download the dataset to that directory.

```.sh
mkdir data
wget -P data/ "https://github.com/anndvision/data/raw/main/jasmin/four_outputs_liqcf_pacific.csv"
```

## Simulated Dose-Response Example

Hyperparameter tuning.

```.sh
overcast \
    tune \
        --job-dir output/ \
        --gpu-per-model 0.25 \
    dose-response \
        --gamma-t 0.3 \
        --gamma-y 0.5 \
        --num-examples 1000 \
        --bootstrap True \
    appended-treatment-nn
```

monitor tuning

```.sh
tensorboard --logdir output/dose-response_n-1000_gt-0.3_gy-0.5_seed-0/appended-treatment-nn/bohb/
```

### Unconfounded, Finite-Data

Train the models.

```.sh
overcast \
    train \
        --job-dir output/ \
        --gpu-per-model 0.25 \
    dose-response \
        --gamma-t 0.0 \
        --gamma-y 0.0 \
        --num-examples 1000 \
        --bootstrap True \
    appended-treatment-nn \
        --dim-hidden 96 \
        --depth 3 \
        --negative-slope 0.05 \
        --dropout-rate 0.04 \
        --learning-rate 0.0015 \
        --batch-size 32 \
        --epochs 300
```

For evaluation, run the jupyter notebook: `examples/dose-response/evaluate_1k.ipynb`

### Confounded, Finite-Data

Train the models.

```.sh
overcast \
    train \
        --job-dir output/ \
        --gpu-per-model 0.25 \
    dose-response \
        --gamma-t 0.3 \
        --gamma-y 0.5 \
        --num-examples 1000 \
        --bootstrap True \
    appended-treatment-nn \
        --dim-hidden 96 \
        --depth 3 \
        --negative-slope 0.05 \
        --dropout-rate 0.04 \
        --learning-rate 0.0015 \
        --batch-size 32 \
        --epochs 300
```

For evaluation, run the jupyter notebook: `examples/dose-response/evaluate_confounded_1k.ipynb`

### Confounded, "Infinite-Data"

Train the models.

```.sh
overcast \
    train \
        --job-dir output/ \
        --gpu-per-model 0.5 \
    dose-response \
        --gamma-t 0.3 \
        --gamma-y 0.5 \
        --num-examples 100000 \
        --bootstrap True \
    appended-treatment-nn \
        --dim-hidden 96 \
        --depth 3 \
        --negative-slope 0.05 \
        --dropout-rate 0.04 \
        --learning-rate 0.0015 \
        --batch-size 32 \
        --epochs 50
```

For evaluation, run the jupyter notebook: `examples/dose-response/evaluate_confounded_100k.ipynb`

## Aerosol-Cloud-Interactions

### Transformer

Train the model

```.sh
overcast \
    train \
        --job-dir output/ \
        --gpu-per-model 0.5 \
    jasmin-daily \
        --root data/four_outputs_liqcf_pacific.csv \
        --bootstrap True \
    appended-treatment-transformer
```

For evaluation, run the jupyter notebook: `overcast/examples/aci/transformer/evaluate.ipynb`

Optional Hyperparameter tuning.

```.sh
overcast \
    tune \
        --job-dir output/ \
        --gpu-per-model 0.25 \
    jasmin-daily \
        --root data/four_outputs_liqcf_pacific.csv \
    appended-treatment-transformer
```

Monitor tuning.

```.sh
tensorboard --logdir output/jasmin-daily_treatment-tot_aod_covariates-RH900-RH850-RH700-LTS-EIS-w500-whoi_sst_outcomes-l_re-liq_pc-cod-cwp_bins-1/appended-treatment-nn/bohb/
```

### Neural Network

Train model.

```.sh
overcast \
    train \
        --job-dir output/ \
        --gpu-per-model 0.5 \
    jasmin \
        --root data/four_outputs_liqcf_pacific.csv \
        --bootstrap True \
    appended-treatment-nn
```

For evaluation, run the jupyter notebook: `overcast/examples/aci/neural_network/evaluate.ipynb`

Optional Hyperparameter tuning.

```.sh
overcast \
    tune \
        --job-dir output/ \
        --gpu-per-model 0.25 \
    jasmin \
        --root data/four_outputs_liqcf_pacific.csv \
    appended-treatment-nn
```

Monitor tuning.

```.sh
tensorboard --logdir output/jasmin_treatment-tot_aod_covariates-RH900-RH850-RH700-LTS-EIS-w500-whoi_sst_outcomes-l_re-liq_pc-cod-cwp_bins-1/appended-treatment-nn/bohb/
```
