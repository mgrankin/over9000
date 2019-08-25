# Optimizers and tests 

Every result is avg of 20 runs.

| Dataset  | Baseline: Adam + OneCycle | Over9000 (RAdam + LARS + Lookahead) | Ralamb (RAdam + LARS) | Ranger (RAdam + Lookahead)| Novograd | Radam | 
| ------------- | ------------- | --|-- | -- | -- | -- |
| Imagenette size 128, 5 epoch | 0.8577  | 0.8746 | 0.8657 | 0.8616 | 0.8724 | 0.8483 |
| Imagewoof size 128, 5 epoch  | 0.6250  | 0.6539 | 0.5851 | 0.6086 | 0.6189 | 0.542 |
