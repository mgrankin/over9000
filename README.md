# Optimizers and tests 

Every result is avg of 20 runs.

Dataset                               | LR Schedule| Imagenette size 128, 5 epoch | Imagewoof size 128, 5 epoch
---                                   | -- | ---                          | ---
Adam - baseline                |OneCycle| 0.8493                       | 0.6125
RangerLars (RAdam + LARS + Lookahead) |Flat and anneal| 0.8732                       | 0.6523
Ralamb (RAdam + LARS)                 |Flat and anneal| 0.8675                       | 0.6367
Ranger (RAdam + Lookahead)            |Flat and anneal| 0.8594                       | 0.5946
Novograd                              |Flat and anneal| 0.8711                       | 0.6126
Radam                                 |Flat and anneal| 0.8444                       | 0.537
Lookahead                             |OneCycle| 0.8578                       | 0.6106
Lamb                                  |OneCycle| 0.8400                       | 0.5597
DiffGrad                              |OneCycle| 0.8527                       | 0.5912
AdaMod                              |OneCycle| 0.8473                       | 0.6132
