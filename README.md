# Kaggle Classification

## Setup

1. Install [TaskFile](https://taskfile.dev/installation/)

1. Install [miniforge](https://github.com/conda-forge/miniforge)

1. Generate [Kaggle API credentials](https://github.com/Kaggle/kaggle-api#api-credentials)

1. Build environment & download all data: `task init`

## Training

`task train -- (name of config e.g. petals_config)`

See [CONFIGS_MAPPING](kaggle_classification/configs/__init__.py) for list of configs
