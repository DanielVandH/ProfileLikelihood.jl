# Optimiser benchmarks
## Basics
The table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and    $n = 300 $ data points.
   - Linear exponential ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y $ with initial condition   $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $\mathrm du/\mathrm dt = \lambda u (1 - u/K)$ with initial condition   $u(0) = u_0  $.    There are four parameters and   $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective (Optimization) | Regression<br>Linear exponential ODE<br>Logistic ODE | 277.394 ns<br>20.900 μs<br>11.100 μs | 289.115 ns<br>22.523 μs<br>12.042 μs | 80 bytes<br>8.08 KiB<br>2.45 KiB | 2<br>13<br>12 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.622 ms<br>7.707 ms<br>6.496 ms | 4.195 ms<br>8.160 ms<br>7.598 ms | 3.52 MiB<br>2.92 MiB<br>1.19 MiB | 8486<br>6940<br>8519 |
| `Optimization.solve` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.589 ms<br>7.649 ms<br>5.983 ms | 3.107 ms<br>8.004 ms<br>6.454 ms | 3.52 MiB<br>2.92 MiB<br>1.19 MiB | 8486<br>6940<br>8519 |
| `LikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 16.132 ns<br>8.500 ns<br>9.100 ns | 16.448 ns<br>8.823 ns<br>9.269 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile` (all variables) | Regression<br>Linear exponential ODE<br>Logistic ODE | 96.161 ms<br>516.764 ms<br>946.971 ms | 98.888 ms<br>520.216 ms<br>950.123 ms | 102.66 MiB<br>199.22 MiB<br>222.56 MiB | 634270<br>861736<br>2708232 |
| | | | | | |
| `profile` (one variable) | Regression<br>Linear exponential ODE<br>Logistic ODE | 22.352 ms<br>220.503 ms<br>177.572 ms | 23.915 ms<br>223.310 ms<br>179.130 ms | 25.13 MiB<br>84.46 MiB<br>42.12 MiB | 150089<br>359831<br>511403 |
| `profile` (one variable, lowered) | Regression<br>Linear exponential ODE<br>Logistic ODE | 21.205 ms<br>239.593 ms<br>178.721 ms | 22.955 ms<br>239.747 ms<br>182.213 ms | 25.13 MiB<br>84.46 MiB<br>42.12 MiB | 150089<br>359831<br>511403 |
| | | | | | |
| `prepare_profile` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.989 μs<br>2.710 μs<br>3.478 μs | 4.574 μs<br>3.145 μs<br>4.176 μs | 6.23 KiB<br>5.09 KiB<br>5.64 KiB | 80<br>50<br>65 |
| `update_prob` (remove bounds) | Regression<br>Linear exponential ODE<br>Logistic ODE | 1.910 μs<br>1.004 μs<br>1.400 μs | 2.169 μs<br>1.189 μs<br>1.437 μs | 1.50 KiB<br>864 bytes<br>1.16 KiB | 46<br>26<br>36 |
| `update_prob` (set initial guess) | Regression<br>Linear exponential ODE<br>Logistic ODE | 6.100 ns<br>3.300 ns<br>4.000 ns | 6.135 ns<br>3.541 ns<br>4.501 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 24.975 ns<br>14.329 ns<br>17.134 ns | 27.148 ns<br>14.990 ns<br>18.343 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `construct_new_f` | Regression<br>Linear exponential ODE<br>Logistic ODE | 18.236 ns<br>11.700 ns<br>13.814 ns | 18.523 ns<br>12.473 ns<br>14.623 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set initial guess and new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 33.266 ns<br>19.840 ns<br>23.972 ns | 34.091 ns<br>20.601 ns<br>25.494 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 31.900 μs<br>4.146 ms<br>4.497 ms | 36.170 μs<br>4.508 ms<br>5.128 ms | 37.66 KiB<br>1.59 MiB<br>1.10 MiB | 242<br>6794<br>13348 |
| `step_profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 506.800 μs<br>3.982 ms<br>3.569 ms | 567.885 μs<br>4.187 ms<br>3.892 ms | 623.58 KiB<br>1.53 MiB<br>922.09 KiB | 3469<br>6498<br>10929 |
| `find_endpoint!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 10.566 ms<br>108.500 ms<br>85.041 ms | 12.143 ms<br>109.997 ms<br>86.646 ms | 12.72 MiB<br>40.65 MiB<br>18.74 MiB | 75905<br>173190<br>227523 |
| | | | | | |
| `confidence_intervals` (all) | Regression<br>Linear exponential ODE<br>Logistic ODE | 26.100 μs<br>17.400 μs<br>29.300 μs | 29.262 μs<br>19.920 μs<br>33.250 μs | 41.41 KiB<br>26.67 KiB<br>45.34 KiB | 70<br>44<br>57 |
| `confidence_intervals` (single) | Regression<br>Linear exponential ODE<br>Logistic ODE | 4.686 μs<br>6.820 μs<br>5.629 μs | 5.361 μs<br>7.791 μs<br>6.580 μs | 7.39 KiB<br>11.06 KiB<br>8.22 KiB | 13<br>13<br>13 |
| | | | | | |
| `ProfileLikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 27.008 ns<br>18.637 ns<br>20.741 ns | 27.973 ns<br>19.474 ns<br>22.476 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
