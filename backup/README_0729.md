# DPO Training and Evaluation

This repository provides scripts and notebooks for training and evaluating models using the Distributionally Preferring Outputs (DPO) framework. Below, you will find descriptions of the key files and directories included in this repository.

## Files

### `dpo_eval.py`
A Python script containing various functions to evaluate metrics, obtain scores, and compute rewards for the models. 

### `dpo_even_token.ipynb`
A Jupyter notebook used to train and evaluate a model where the reward is based on the "number of even tokens." The model is trained using the DPO framework.
- Output to `output` and `model_output` directories. For more information see ***Directories*** Section. 
### `dpo_length.ipynb`
A Jupyter notebook used to train and evaluate a model where the reward is based on the "number of tokens." The model is trained using the DPO framework.
- Output to `output` and `model_output` directories. For more information see ***Directories*** Section. 
### `dpo_mos.ipynb`
A Jupyter notebook used to train and evaluate a model using the MOSNET score acquired from NISQA as the reward. The model is trained using the DPO framework.
- Output to `output` and `model_output` directories. For more information see ***Directories*** Section. 
### `quick_train_eval.ipynb`
A Jupyter notebook that serves two main purposes:
1. Continues training from a saved checkpoint.
2. Evaluates the MOS score of any trained model checkpoint.
### `test_audio_metrics.ipynb`
A Jupyter notebook that test out different audio metrics
## Directories

### `model_output/{ts}`
This directory contains:
- `log_training.log`: A log file detailing the entire training process, evaluation results, and other relevant information.
- `iter_0`, `iter_1`, ..., etc.: Subdirectories representing each iteration of the training process. Each subdirectory contains files related to that specific training iteration.
- `{ts}` represents the timestamp of each training session, formatted as `MMDD-HHMM` (e.g., `0728-0128`).

### `output/{ts}`
This directory contains:
- Audio outputs generated by the model.
- `data_iter_0.json`, `data_iter_1.json`, ..., etc.: Files containing the chosen and rejected data for each training iteration.
- `{ts}` represents the timestamp of each training session, formatted as `MMDD-HHMM` (e.g., `0728-0128`).
