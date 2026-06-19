| name | lr | valid PPL | test PPL | #params | s |
|---|---|---|---|---|---|
| partA_arch__d_model384__num_layers6 | 0.0005 | 38.20 | - | 47326033 | 929.5 |
| partA_arch__d_model256__num_layers6 | 0.0005 | 38.39 | - | 30783057 | 895.2 |
| partA_arch__d_model384__num_layers4 | 0.0005 | 38.86 | - | 44564561 | 862.1 |
| partA_arch__d_model256__num_layers4 | 0.0005 | 39.46 | - | 29203537 | 848.0 |
| partA_baseline__lr0.0005 | 0.0005 | 39.46 | - | 29203537 | 839.2 |
| partA_baseline__lr0.001 | 0.001 | 39.65 | - | 29203537 | 749.5 |
| partA_baseline__lr0.0001 | 0.0001 | 40.98 | - | 29203537 | 1963.7 |
| smoke | 0.001 | 6835.52 | - | 3279409 | 100.5 |
