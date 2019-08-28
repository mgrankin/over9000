# Optimizers and tests 

LARS family of optimizers are more hungry to GPU time pre batch, so my initial comparison isn't fare. No SOTA today, sorry.

The current implementaions of Ralamb and RangerLars/Over9000 probably contains a bug, will update on that soon.

Every result is avg of 20 runs.

| Dataset  | Baseline: Adam + OneCycle | RangerLars aka Over9000 (RAdam + LARS + Lookahead) | Ralamb (RAdam + LARS) | Ranger (RAdam + Lookahead)| Novograd | Radam | Lookahead |
| ------------- | ------------- | --|-- | -- | -- | -- | -- |
| Imagenette size 128, 5 epoch | 0.8493  | 0.8755 | 0.8621 | 0.8594 | 0.8711 | 0.8444 | 0.8578 |
| Imagewoof size 128, 5 epoch  | 0.6125  | 0.6451 | 0.5737 | 0.5946 | 0.6126 | 0.537 | 0.6106 |
