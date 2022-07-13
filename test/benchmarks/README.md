# Optimiser benchmarks
## Basics
The table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and    $n = 300 $ data points.
   - Linear exponential ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y $ with initial condition   $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $\mathrm du/\mathrm dt = \lambda u (1 - u/K)$ with initial condition   $u(0) = u_0  $.    There are four parameters and   $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective (Optimization) | Regression<br>Linear exponential ODE<br>Logistic ODE | 293.051 ns<br>19.900 μs<br>10.900 μs | 316.805 ns<br>21.787 μs<br>12.005 μs | 80 bytes<br>8.08 KiB<br>2.45 KiB | 2<br>13<br>12 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.945 ms<br>8.163 ms<br>5.962 ms | 4.392 ms<br>8.843 ms<br>6.460 ms | 3.50 MiB<br>3.03 MiB<br>1.20 MiB | 7894<br>7205<br>8591 |
| `Optimization.solve` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.414 ms<br>7.921 ms<br>5.830 ms | 3.521 ms<br>8.548 ms<br>6.313 ms | 3.50 MiB<br>3.03 MiB<br>1.20 MiB | 7894<br>7205<br>8591 |
| `LikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 22.166 ns<br>10.800 ns<br>10.700 ns | 23.178 ns<br>12.527 ns<br>11.250 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile` (all variables) | Regression<br>Linear exponential ODE<br>Logistic ODE | 104.409 ms<br>525.346 ms<br>959.048 ms | 103.741 ms<br>526.149 ms<br>957.064 ms | 101.74 MiB<br>198.69 MiB<br>220.20 MiB | 615631<br>857493<br>2680643 |
| | | | | | |
| `profile` (one variable) | Regression<br>Linear exponential ODE<br>Logistic ODE | 23.246 ms<br>241.831 ms<br>181.088 ms | 24.880 ms<br>240.828 ms<br>181.338 ms | 24.57 MiB<br>84.82 MiB<br>41.44 MiB | 143517<br>360645<br>503121 |
| `profile` (one variable, lowered) | Regression<br>Linear exponential ODE<br>Logistic ODE | 23.222 ms<br>243.961 ms<br>174.831 ms | 24.760 ms<br>243.531 ms<br>178.305 ms | 24.57 MiB<br>84.82 MiB<br>41.44 MiB | 143517<br>360645<br>503121 |
| | | | | | |
| `prepare_profile` | Regression<br>Linear exponential ODE<br>Logistic ODE | 4.517 μs<br>2.950 μs<br>3.511 μs | 5.746 μs<br>3.462 μs<br>3.997 μs | 6.23 KiB<br>5.09 KiB<br>5.64 KiB | 80<br>50<br>65 |
| `update_prob` (remove bounds) | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.110 μs<br>937.755 ns<br>1.360 μs | 2.499 μs<br>1.113 μs<br>1.435 μs | 1.50 KiB<br>864 bytes<br>1.16 KiB | 46<br>26<br>36 |
| `update_prob` (set initial guess) | Regression<br>Linear exponential ODE<br>Logistic ODE | 6.300 ns<br>4.100 ns<br>4.200 ns | 6.913 ns<br>4.411 ns<br>4.314 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 25.276 ns<br>17.317 ns<br>17.335 ns | 27.947 ns<br>19.064 ns<br>18.889 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `construct_new_f` | Regression<br>Linear exponential ODE<br>Logistic ODE | 18.437 ns<br>14.000 ns<br>14.014 ns | 19.294 ns<br>14.861 ns<br>14.380 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set initial guess and new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 33.534 ns<br>24.248 ns<br>24.297 ns | 35.164 ns<br>25.575 ns<br>25.154 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 34.800 μs<br>4.138 ms<br>3.946 ms | 40.871 μs<br>4.515 ms<br>4.278 ms | 36.92 KiB<br>1.69 MiB<br>1.05 MiB | 237<br>7164<br>12733 |
| `step_profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 575.000 μs<br>3.898 ms<br>3.402 ms | 653.032 μs<br>4.405 ms<br>3.706 ms | 604.70 KiB<br>1.51 MiB<br>918.64 KiB | 3324<br>6424<br>10888 |
| `find_endpoint!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 11.963 ms<br>107.094 ms<br>78.537 ms | 13.134 ms<br>107.972 ms<br>81.412 ms | 12.28 MiB<br>40.40 MiB<br>18.63 MiB | 71700<br>171784<br>226129 |
| | | | | | |
| `confidence_intervals` (all) | Regression<br>Linear exponential ODE<br>Logistic ODE | 28.200 μs<br>18.800 μs<br>30.900 μs | 31.763 μs<br>20.711 μs<br>34.157 μs | 41.41 KiB<br>26.67 KiB<br>45.34 KiB | 70<br>44<br>57 |
| `confidence_intervals` (single) | Regression<br>Linear exponential ODE<br>Logistic ODE | 5.186 μs<br>8.040 μs<br>5.450 μs | 5.896 μs<br>9.584 μs<br>6.075 μs | 7.39 KiB<br>11.06 KiB<br>8.22 KiB | 13<br>13<br>13 |
| | | | | | |
| `ProfileLikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 37.563 ns<br>22.668 ns<br>22.700 ns | 40.164 ns<br>23.793 ns<br>24.341 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
