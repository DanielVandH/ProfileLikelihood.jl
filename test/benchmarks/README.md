# Optimiser benchmarks
## Basics
The table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and    $n = 300 $ data points.
   - Linear exponential ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y $ with initial condition   $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $\mathrm du/\mathrm dt = \lambda u (1 - u/K)$ with initial condition   $u(0) = u_0  $.    There are four parameters and   $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective (Optimization) | Regression<br>Linear exponential ODE<br>Logistic ODE | 298.142 ns<br>19.500 μs<br>10.600 μs | 341.414 ns<br>20.697 μs<br>11.691 μs | 80 bytes<br>8.08 KiB<br>2.45 KiB | 2<br>13<br>12 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.905 ms<br>7.886 ms<br>5.738 ms | 3.297 ms<br>8.328 ms<br>6.184 ms | 3.50 MiB<br>3.03 MiB<br>1.20 MiB | 7894<br>7205<br>8591 |
| `Optimization.solve` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.579 ms<br>7.679 ms<br>5.685 ms | 3.007 ms<br>8.130 ms<br>6.097 ms | 3.50 MiB<br>3.03 MiB<br>1.20 MiB | 7894<br>7205<br>8591 |
| `LikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 21.600 ns<br>10.700 ns<br>10.700 ns | 22.991 ns<br>11.524 ns<br>10.986 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile` (all variables) | Regression<br>Linear exponential ODE<br>Logistic ODE | 101.065 ms<br>517.455 ms<br>903.836 ms | 100.235 ms<br>519.107 ms<br>908.571 ms | 101.74 MiB<br>198.69 MiB<br>220.20 MiB | 615631<br>857493<br>2680643 |
| | | | | | |
| `profile` (one variable) | Regression<br>Linear exponential ODE<br>Logistic ODE | 22.496 ms<br>241.231 ms<br>175.855 ms | 24.191 ms<br>245.263 ms<br>177.924 ms | 24.57 MiB<br>84.82 MiB<br>41.44 MiB | 143517<br>360645<br>503121 |
| `profile` (one variable, lowered) | Regression<br>Linear exponential ODE<br>Logistic ODE | 22.691 ms<br>234.213 ms<br>177.832 ms | 24.528 ms<br>235.105 ms<br>179.964 ms | 24.57 MiB<br>84.82 MiB<br>41.44 MiB | 143517<br>360645<br>503121 |
| | | | | | |
| `prepare_profile` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.778 μs<br>2.720 μs<br>3.444 μs | 4.454 μs<br>3.103 μs<br>3.976 μs | 6.23 KiB<br>5.09 KiB<br>5.64 KiB | 80<br>50<br>65 |
| `update_prob` (remove bounds) | Regression<br>Linear exponential ODE<br>Logistic ODE | 1.880 μs<br>914.130 ns<br>1.370 μs | 2.201 μs<br>1.114 μs<br>1.469 μs | 1.50 KiB<br>864 bytes<br>1.16 KiB | 46<br>26<br>36 |
| `update_prob` (set initial guess) | Regression<br>Linear exponential ODE<br>Logistic ODE | 6.200 ns<br>4.100 ns<br>4.200 ns | 6.575 ns<br>4.590 ns<br>4.392 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 24.598 ns<br>17.234 ns<br>17.234 ns | 26.826 ns<br>18.331 ns<br>18.393 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `construct_new_f` | Regression<br>Linear exponential ODE<br>Logistic ODE | 18.418 ns<br>14.000 ns<br>14.000 ns | 18.895 ns<br>15.451 ns<br>14.973 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set initial guess and new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 32.797 ns<br>24.273 ns<br>23.671 ns | 34.588 ns<br>26.097 ns<br>25.400 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 34.100 μs<br>4.350 ms<br>3.943 ms | 39.800 μs<br>4.734 ms<br>4.220 ms | 36.92 KiB<br>1.69 MiB<br>1.05 MiB | 237<br>7164<br>12733 |
| `step_profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 559.600 μs<br>4.032 ms<br>3.385 ms | 633.787 μs<br>4.329 ms<br>3.688 ms | 604.70 KiB<br>1.51 MiB<br>918.64 KiB | 3324<br>6424<br>10888 |
| `find_endpoint!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 11.169 ms<br>111.531 ms<br>79.525 ms | 12.063 ms<br>113.748 ms<br>82.112 ms | 12.28 MiB<br>40.40 MiB<br>18.63 MiB | 71700<br>171784<br>226129 |
| | | | | | |
| `confidence_intervals` (all) | Regression<br>Linear exponential ODE<br>Logistic ODE | 27.200 μs<br>19.500 μs<br>30.800 μs | 30.378 μs<br>21.449 μs<br>34.735 μs | 41.41 KiB<br>26.67 KiB<br>45.34 KiB | 70<br>44<br>57 |
| `confidence_intervals` (single) | Regression<br>Linear exponential ODE<br>Logistic ODE | 5.100 μs<br>7.420 μs<br>5.329 μs | 5.655 μs<br>8.228 μs<br>6.059 μs | 7.39 KiB<br>11.06 KiB<br>8.22 KiB | 13<br>13<br>13 |
| | | | | | |
| `ProfileLikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 37.450 ns<br>22.668 ns<br>22.645 ns | 39.446 ns<br>23.536 ns<br>23.549 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
