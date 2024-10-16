### Entry point
`dpo_claps.ipynb`

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