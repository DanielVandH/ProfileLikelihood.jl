# Optimiser benchmarks
## BasicsThe table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and    $n = 300 $ data points.
   - Linear exponential ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y $ with initial condition   $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y (1 - u/K)$ with initial condition   $y(0) = y_0  $.    There are four parameters and   $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective (Optimization) | Regression<br>Linear exponential ODE<br>Logistic ODE | 280.211 ns<br>20.300 μs<br>10.900 μs | 323.423 ns<br>22.524 μs<br>11.669 μs | 80 bytes<br>8.08 KiB<br>2.45 KiB | 2<br>13<br>12 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.605 ms<br>7.927 ms<br>5.920 ms | 2.998 ms<br>8.566 ms<br>6.895 ms | 3.52 MiB<br>2.92 MiB<br>1.19 MiB | 8486<br>6939<br>8519 |
| `Optimization.solve` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.551 ms<br>7.741 ms<br>5.847 ms | 2.894 ms<br>8.335 ms<br>6.256 ms | 3.52 MiB<br>2.92 MiB<br>1.19 MiB | 8486<br>6939<br>8519 |
| `LikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 16.116 ns<br>8.700 ns<br>9.100 ns | 16.758 ns<br>9.210 ns<br>9.169 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile` (all variables) | Regression<br>Linear exponential ODE<br>Logistic ODE | 95.027 ms<br>546.153 ms<br>901.599 ms | 94.391 ms<br>547.596 ms<br>910.747 ms | 102.66 MiB<br>199.22 MiB<br>222.56 MiB | 634270<br>861736<br>2708232 |
| | | | | | |
| `profile` (one variable) | Regression<br>Linear exponential ODE<br>Logistic ODE | 21.554 ms<br>231.597 ms<br>170.172 ms | 23.616 ms<br>239.927 ms<br>172.231 ms | 25.13 MiB<br>84.46 MiB<br>42.12 MiB | 150089<br>359831<br>511403 |
| `profile` (one variable, lowered) | Regression<br>Linear exponential ODE<br>Logistic ODE | 21.796 ms<br>240.890 ms<br>168.231 ms | 23.785 ms<br>256.122 ms<br>171.099 ms | 25.13 MiB<br>84.46 MiB<br>42.12 MiB | 150089<br>359831<br>511403 |
| | | | | | |
| `prepare_profile` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.922 μs<br>2.870 μs<br>3.467 μs | 4.585 μs<br>3.383 μs<br>3.840 μs | 6.23 KiB<br>5.09 KiB<br>5.64 KiB | 80<br>50<br>65 |
| `update_prob` (remove bounds) | Regression<br>Linear exponential ODE<br>Logistic ODE | 1.930 μs<br>992.593 ns<br>1.410 μs | 2.086 μs<br>1.087 μs<br>1.441 μs | 1.50 KiB<br>864 bytes<br>1.16 KiB | 46<br>26<br>36 |
| `update_prob` (set initial guess) | Regression<br>Linear exponential ODE<br>Logistic ODE | 6.100 ns<br>3.300 ns<br>4.000 ns | 6.153 ns<br>3.671 ns<br>4.036 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 25.000 ns<br>14.700 ns<br>17.117 ns | 25.948 ns<br>15.456 ns<br>17.535 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `construct_new_f` | Regression<br>Linear exponential ODE<br>Logistic ODE | 18.236 ns<br>11.700 ns<br>13.814 ns | 20.087 ns<br>12.129 ns<br>14.855 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set initial guess and new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 42.369 ns<br>19.800 ns<br>23.972 ns | 44.312 ns<br>20.273 ns<br>24.679 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 36.900 μs<br>3.895 ms<br>4.384 ms | 43.999 μs<br>4.206 ms<br>4.560 ms | 37.66 KiB<br>1.59 MiB<br>1.10 MiB | 242<br>6794<br>13348 |
| `step_profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 525.200 μs<br>3.963 ms<br>3.656 ms | 646.179 μs<br>4.314 ms<br>3.860 ms | 591.67 KiB<br>1.53 MiB<br>922.09 KiB | 3309<br>6498<br>10929 |
| `find_endpoint!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 12.450 ms<br>112.943 ms<br>89.953 ms | 14.038 ms<br>113.370 ms<br>95.666 ms | 12.72 MiB<br>40.65 MiB<br>18.85 MiB | 75905<br>173190<br>228835 |
| | | | | | |
| `confidence_intervals` (all) | Regression<br>Linear exponential ODE<br>Logistic ODE | 30.200 μs<br>18.100 μs<br>29.800 μs | 35.127 μs<br>19.848 μs<br>33.775 μs | 41.41 KiB<br>26.67 KiB<br>45.34 KiB | 70<br>44<br>57 |
| `confidence_intervals` (single) | Regression<br>Linear exponential ODE<br>Logistic ODE | 5.157 μs<br>7.000 μs<br>5.229 μs | 5.850 μs<br>7.809 μs<br>6.018 μs | 7.39 KiB<br>11.06 KiB<br>8.22 KiB | 13<br>13<br>13 |
| | | | | | |
| `ProfileLikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 27.410 ns<br>18.700 ns<br>20.762 ns | 31.026 ns<br>19.445 ns<br>21.575 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
