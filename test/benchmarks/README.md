# Optimiser benchmarks
## Basics
The table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and    $n = 300 $ data points.
   - Linear exponential ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y $ with initial condition   $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $\mathrm du/\mathrm dt = \lambda u (1 - u/K)$ with initial condition   $u(0) = u_0  $.    There are four parameters and   $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective (Optimization) | Regression<br>Linear exponential ODE<br>Logistic ODE | 277.411 ns<br>20.800 μs<br>12.000 μs | 307.442 ns<br>22.319 μs<br>13.574 μs | 80 bytes<br>8.08 KiB<br>2.45 KiB | 2<br>13<br>12 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.699 ms<br>7.905 ms<br>5.785 ms | 3.236 ms<br>8.469 ms<br>6.293 ms | 3.52 MiB<br>2.92 MiB<br>1.19 MiB | 8486<br>6939<br>8519 |
| `Optimization.solve` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.669 ms<br>7.505 ms<br>5.857 ms | 3.114 ms<br>7.857 ms<br>6.498 ms | 3.52 MiB<br>2.92 MiB<br>1.19 MiB | 8486<br>6939<br>8519 |
| `LikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 16.333 ns<br>8.600 ns<br>9.100 ns | 17.392 ns<br>9.060 ns<br>9.219 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile` (all variables) | Regression<br>Linear exponential ODE<br>Logistic ODE | 93.852 ms<br>547.653 ms<br>978.773 ms | 94.092 ms<br>546.768 ms<br>978.566 ms | 102.66 MiB<br>199.22 MiB<br>222.56 MiB | 634270<br>861736<br>2708232 |
| | | | | | |
| `profile` (one variable) | Regression<br>Linear exponential ODE<br>Logistic ODE | 21.081 ms<br>218.980 ms<br>175.697 ms | 23.167 ms<br>224.314 ms<br>176.780 ms | 25.13 MiB<br>84.46 MiB<br>42.12 MiB | 150089<br>359831<br>511403 |
| `profile` (one variable, lowered) | Regression<br>Linear exponential ODE<br>Logistic ODE | 21.215 ms<br>219.993 ms<br>173.323 ms | 23.283 ms<br>223.020 ms<br>177.261 ms | 25.13 MiB<br>84.46 MiB<br>42.12 MiB | 150089<br>359831<br>511403 |
| | | | | | |
| `prepare_profile` | Regression<br>Linear exponential ODE<br>Logistic ODE | 4.256 μs<br>3.170 μs<br>3.556 μs | 4.755 μs<br>4.165 μs<br>4.050 μs | 6.23 KiB<br>5.09 KiB<br>5.64 KiB | 80<br>50<br>65 |
| `update_prob` (remove bounds) | Regression<br>Linear exponential ODE<br>Logistic ODE | 1.960 μs<br>1.097 μs<br>1.530 μs | 2.003 μs<br>1.258 μs<br>1.572 μs | 1.50 KiB<br>864 bytes<br>1.16 KiB | 46<br>26<br>36 |
| `update_prob` (set initial guess) | Regression<br>Linear exponential ODE<br>Logistic ODE | 6.100 ns<br>3.300 ns<br>4.000 ns | 6.121 ns<br>3.389 ns<br>4.088 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 24.373 ns<br>14.700 ns<br>17.034 ns | 26.899 ns<br>15.595 ns<br>17.309 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `construct_new_f` | Regression<br>Linear exponential ODE<br>Logistic ODE | 17.718 ns<br>11.700 ns<br>13.800 ns | 18.219 ns<br>11.917 ns<br>14.073 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set initial guess and new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 32.530 ns<br>19.820 ns<br>24.072 ns | 34.025 ns<br>20.348 ns<br>25.162 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 33.500 μs<br>3.855 ms<br>4.284 ms | 38.164 μs<br>4.217 ms<br>4.634 ms | 37.66 KiB<br>1.59 MiB<br>1.10 MiB | 242<br>6794<br>13348 |
| `step_profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 521.100 μs<br>3.908 ms<br>3.510 ms | 594.156 μs<br>4.257 ms<br>3.765 ms | 623.58 KiB<br>1.53 MiB<br>922.09 KiB | 3469<br>6498<br>10929 |
| `find_endpoint!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 10.586 ms<br>111.379 ms<br>82.568 ms | 11.830 ms<br>111.645 ms<br>87.390 ms | 12.72 MiB<br>40.65 MiB<br>18.74 MiB | 75905<br>173190<br>227523 |
| | | | | | |
| `confidence_intervals` (all) | Regression<br>Linear exponential ODE<br>Logistic ODE | 26.800 μs<br>18.300 μs<br>29.300 μs | 30.116 μs<br>20.265 μs<br>32.770 μs | 41.41 KiB<br>26.67 KiB<br>45.34 KiB | 70<br>44<br>57 |
| `confidence_intervals` (single) | Regression<br>Linear exponential ODE<br>Logistic ODE | 4.814 μs<br>6.960 μs<br>5.386 μs | 5.492 μs<br>7.950 μs<br>5.963 μs | 7.39 KiB<br>11.06 KiB<br>8.22 KiB | 13<br>13<br>13 |
| | | | | | |
| `ProfileLikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 27.209 ns<br>18.656 ns<br>20.942 ns | 27.580 ns<br>19.345 ns<br>24.214 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
