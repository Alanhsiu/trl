### Entry point
* `dpo_claps.ipynb` (base)
* `dpo_claps_optimized_io.ipynb` (optimized IO version)
* `dpo_claps_mos` (Claps + MOS version)

related files:
`dpo_eval.py`, `vc/`

### Current status
CLAPS: input 

### Some actions
Don't save the model output
```
(trl) (base) b0990106x@whhwh4ctr1728964608578-hh54w:/work/b0990106x/trl$ du -sh model_output/
666G    model_output/
```
### Modify IO part
1. avoid loading checkpoint repeatedly
2. avoid saving model output

### Plot
1. std error
2. max and min

### Exp Todos
1. train / test set difference (5 fixed / 5 fixed), eval freq = 5
2. train / test set difference (5 random / 5 fixed), eval freq = 5