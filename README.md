# Optimizers and tests 

LARS family of optimizers are more hungry to GPU time pre batch, so my initial comparison isn't fare. No SOTA today, sorry.

Every result is avg of 20 runs.

| Dataset  | Baseline: Adam + OneCycle | RangerLars (RAdam + LARS + Lookahead) | Ralamb (RAdam + LARS) | Ranger (RAdam + Lookahead)| Novograd | Radam | Lookahead | Lamb | 
| ------------- | ------------- | --|-- | -- | -- | -- | -- | -- | 
| Imagenette size 128, 5 epoch | 0.8493  | 0.8732 | 0.8675 | 0.8594 | 0.8711 | 0.8444 | 0.8578 | 0.8400 |
| Imagewoof size 128, 5 epoch  | 0.6125  | 0.6523 | 0.6367 | 0.5946 | 0.6126 | 0.537 | 0.6106 | 0.5597 |
