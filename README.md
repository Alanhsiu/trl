### Entry point
* `dpo_claps.ipynb` (base) -> backup on 11/1
* `dpo_claps_optimized_io.ipynb` (optimized IO version) -> rename to `dpo_claps.ipynb` on 11/1
* `dpo_claps_mos`
* `dpo_mos_nisqa` 
* `dpo_mos_ut` 
* `dpo_asr` 
* `dpo_claps_asr`

related files:
`dpo_eval.py`, `vc/`

### Current status
CLAPS: input 

### Notification
Don't save the model output
```
(trl) (base) b0990106x@whhwh4ctr1728964608578-hh54w:/work/b0990106x/trl$ du -sh model_output/
666G    model_output/
```

### Data Source
[soxdata_encodec](https://huggingface.co/datasets/lca0503/soxdata_encodec/viewer/default/test)
Evaluation: eval_train_indices: [8, 16, 65, 100, 102]
Evaluation: eval_test_indices: [105, 112, 132, 140, 184]

### Pre-Task
```
sudo apt-get update && sudo apt-get install tmux
```
```
pip install nbconvert
jupyter nbconvert --to script <name>.ipynb
```

### Modify IO part
1. avoid loading checkpoint repeatedly
2. avoid saving model output

### Plot
1. std error
2. max and min

### Exp Todos
1. CLaps: train / test set difference (5 fixed / 5 fixed), eval freq = 5
2. CLaps+MOS: train / test set difference (5 fixed / 5 fixed), eval freq = 5

### Notes
1. use another mos reward 
2. ut mos reward