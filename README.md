### Entry point
* `dpo_claps_asr.ipynb`

* related files: `dpo_eval.py`, `vc/`, `CLAPS/`

### Data Source
[soxdata_encodec](https://huggingface.co/datasets/lca0503/soxdata_encodec/viewer/default/test)

### Use TMUX to run the script in the background
```
sudo apt-get update && sudo apt-get install tmux
pip install nbconvert
jupyter nbconvert --to script dpo_claps_asr.ipynb.ipynb
```
(alternatively, in VSCode, press `ctrl` + `shift` + `p` and click `Jupyter: Export to Python Script`)

### Use plot.ipynb to plot the result
* be sure to modify the  `model_output_dir`
(it is ok to use tensorboard to visualize the result, however, the plot.ipynb is more flexible and better for report)

### Note
* Reward function is of the form `reward = k*(CLAPS_reward) + (1-k)*(ASR_reward)`, where `k` is a hyperparameter.

### Tensorboard Usage
```
tensorboard --logdir="model_output/1210-2215/tensorboard_logs"
```