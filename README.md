# overcast

Scalable Sensitivity and Uncertainty Analysis for Causal-Effect Estimates of Continuous-Valued Interventions

## Installation

```.sh
git clone git@github.com:anon/overcast.git
cd overcast
conda env create -f environment.yml
conda activate overcast
pip install .
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
