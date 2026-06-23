# NPDP

NPDP is a unified framework for predicting a normalized patent-value score from patent text. It supports LLaMA, Qwen2, GLM, Mistral, and Baichuan with the same regression head, loss implementation, dataset interface, LoRA configuration, and evaluation pipeline.

## Model logic

For every backbone, NPDP applies a one-dimensional linear head to every token, selects the final non-padding token, and maps the result to the configured output range:

```text
patent title + abstract
        -> decoder-only language model
        -> final valid token representation
        -> linear score head
        -> sigmoid (default)
        -> normalized patent value in [0, 1]
```

The default objective is mean squared error between the prediction and the normalized target. `l1`, `smooth_l1`, and `bce` are also supported.

## Project structure

```text
llama_for_npdp/       LLaMA model, fine-tuning, and testing
qwen_for_npdp/        Qwen2 model, fine-tuning, and testing
glm_for_npdp/         GLM model, fine-tuning, and testing
mistral_for_npdp/     Mistral model, fine-tuning, and testing
baichuan_for_npdp/    Baichuan model, fine-tuning, and testing
npdp_model_utils.py   shared head, pooling, activation, and loss
npdp_dataset.py       shared patent CSV dataset
npdp_registry.py      model registry and loading rules
npdp_finetune_common.py
npdp_test_common.py
```

There are no `NAIP`, `NAID`, or `*_for_naipv` imports. Model-specific entry scripts only select a registry entry; all training and evaluation behavior remains centralized.

## Data format

The default CSV columns are:

```csv
title,abstract,target
Patent title,Patent abstract,0.73
```

Column names can be changed with `--title_column`, `--abstract_column`, and `--label_column`.

## Installation

```bash
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build before enabling 4-bit or 8-bit quantization.

## Training

Qwen example:

```bash
python -m qwen_for_npdp.finetune \
  --checkpoint Qwen/Qwen2.5-7B \
  --data_path data/patents_train.csv \
  --label_column target \
  --load_in_4bit \
  --bf16
```

The other entry points are:

```text
python -m llama_for_npdp.finetune
python -m glm_for_npdp.finetune
python -m mistral_for_npdp.finetune
python -m baichuan_for_npdp.finetune
```

For Baichuan, the registry defaults to the fused attention projection `W_pack`. Override `--target_modules` if the chosen checkpoint uses different module names.

## Evaluation

```bash
python -m qwen_for_npdp.test \
  --checkpoint Qwen/Qwen2.5-7B \
  --model_path runs/qwen/<run>/last \
  --data_path data/patents_test.csv \
  --output_csv predictions.csv \
  --bf16
```

The reported metrics are MSE, MAE, and NDCG@20. Model outputs already contain the activated 0-1 prediction, so evaluation must not apply a second sigmoid.
