# Optimiser benchmarks
## Basics
The table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and    $n = 300 $ data points.
   - Linear exponential ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y $ with initial condition   $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $\mathrm du/\mathrm dt = \lambda u (1 - u/K)$ with initial condition   $u(0) = u_0  $.    There are four parameters and   $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective (Optimization) | Regression<br>Linear exponential ODE<br>Logistic ODE | 275.798 ns<br>21.100 μs<br>11.200 μs | 293.523 ns<br>22.871 μs<br>12.383 μs | 80 bytes<br>8.08 KiB<br>2.45 KiB | 2<br>13<br>12 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.636 ms<br>8.067 ms<br>6.107 ms | 2.975 ms<br>8.725 ms<br>6.365 ms | 3.52 MiB<br>2.92 MiB<br>1.19 MiB | 8486<br>6939<br>8519 |
| `Optimization.solve` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.605 ms<br>7.656 ms<br>6.117 ms | 2.955 ms<br>8.112 ms<br>6.500 ms | 3.52 MiB<br>2.92 MiB<br>1.19 MiB | 8486<br>6939<br>8519 |
| `LikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 16.132 ns<br>9.300 ns<br>9.100 ns | 16.504 ns<br>9.615 ns<br>9.219 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile` (all variables) | Regression<br>Linear exponential ODE<br>Logistic ODE | 95.146 ms<br>526.045 ms<br>921.564 ms | 94.557 ms<br>526.853 ms<br>914.379 ms | 102.66 MiB<br>199.22 MiB<br>222.56 MiB | 634270<br>861736<br>2708232 |
| | | | | | |
| `profile` (one variable) | Regression<br>Linear exponential ODE<br>Logistic ODE | 22.263 ms<br>233.533 ms<br>174.622 ms | 24.773 ms<br>237.571 ms<br>176.527 ms | 25.13 MiB<br>84.46 MiB<br>42.12 MiB | 150089<br>359831<br>511403 |
| `profile` (one variable, lowered) | Regression<br>Linear exponential ODE<br>Logistic ODE | 21.149 ms<br>225.848 ms<br>171.842 ms | 22.944 ms<br>227.543 ms<br>175.629 ms | 25.13 MiB<br>84.46 MiB<br>42.12 MiB | 150089<br>359831<br>511403 |
| | | | | | |
| `prepare_profile` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.856 μs<br>2.840 μs<br>3.489 μs | 4.445 μs<br>2.974 μs<br>3.933 μs | 6.23 KiB<br>5.09 KiB<br>5.64 KiB | 80<br>50<br>65 |
| `update_prob` (remove bounds) | Regression<br>Linear exponential ODE<br>Logistic ODE | 1.910 μs<br>893.750 ns<br>1.520 μs | 1.926 μs<br>1.081 μs<br>1.674 μs | 1.50 KiB<br>864 bytes<br>1.16 KiB | 46<br>26<br>36 |
| `update_prob` (set initial guess) | Regression<br>Linear exponential ODE<br>Logistic ODE | 5.900 ns<br>3.300 ns<br>4.000 ns | 6.075 ns<br>3.312 ns<br>4.032 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 24.398 ns<br>14.715 ns<br>17.034 ns | 24.887 ns<br>16.044 ns<br>17.261 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `construct_new_f` | Regression<br>Linear exponential ODE<br>Logistic ODE | 17.800 ns<br>11.800 ns<br>13.814 ns | 18.900 ns<br>13.587 ns<br>14.179 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set initial guess and new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 32.530 ns<br>20.040 ns<br>23.972 ns | 33.721 ns<br>26.893 ns<br>24.778 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 35.800 μs<br>4.099 ms<br>4.240 ms | 40.396 μs<br>4.417 ms<br>4.425 ms | 37.66 KiB<br>1.59 MiB<br>1.10 MiB | 242<br>6794<br>13348 |
| `step_profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 525.100 μs<br>4.005 ms<br>3.536 ms | 592.134 μs<br>4.186 ms<br>3.803 ms | 607.62 KiB<br>1.53 MiB<br>922.09 KiB | 3389<br>6498<br>10929 |
| `find_endpoint!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 10.853 ms<br>101.310 ms<br>84.912 ms | 11.932 ms<br>103.994 ms<br>87.876 ms | 12.72 MiB<br>40.65 MiB<br>18.85 MiB | 75905<br>173190<br>228835 |
| | | | | | |
| `confidence_intervals` (all) | Regression<br>Linear exponential ODE<br>Logistic ODE | 26.900 μs<br>18.000 μs<br>29.950 μs | 30.285 μs<br>20.255 μs<br>33.185 μs | 41.41 KiB<br>26.67 KiB<br>45.34 KiB | 70<br>44<br>57 |
| `confidence_intervals` (single) | Regression<br>Linear exponential ODE<br>Logistic ODE | 4.943 μs<br>6.880 μs<br>5.286 μs | 5.434 μs<br>7.782 μs<br>5.880 μs | 7.39 KiB<br>11.06 KiB<br>8.22 KiB | 13<br>13<br>13 |
| | | | | | |
| `ProfileLikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 27.610 ns<br>18.236 ns<br>20.900 ns | 28.679 ns<br>18.619 ns<br>24.261 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
