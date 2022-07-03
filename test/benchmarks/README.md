# Optimiser benchmarks
## Basics
The table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and    $n = 300 $ data points.
   - Linear exponential ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y $ with initial condition   $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $\mathrm du/\mathrm dt = \lambda u (1 - u/K)$ with initial condition   $u(0) = u_0  $.    There are four parameters and   $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective (Optimization) | Regression<br>Linear exponential ODE<br>Logistic ODE | 271.543 ns<br>68.200 μs<br>61.400 μs | 283.242 ns<br>69.357 μs<br>70.879 μs | 80 bytes<br>31.38 KiB<br>29.81 KiB | 2<br>1431<br>1619 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.511 ms<br>23.757 ms<br>33.869 ms | 2.864 ms<br>24.392 ms<br>37.468 ms | 3.52 MiB<br>9.63 MiB<br>14.56 MiB | 8486<br>417012<br>811132 |
| `Optimization.solve` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.830 ms<br>23.357 ms<br>32.032 ms | 3.133 ms<br>23.727 ms<br>32.920 ms | 3.52 MiB<br>9.63 MiB<br>14.56 MiB | 8486<br>417012<br>811132 |
| `LikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 15.731 ns<br>8.600 ns<br>9.100 ns | 16.701 ns<br>8.706 ns<br>9.154 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile` (all variables) | Regression<br>Linear exponential ODE<br>Logistic ODE | 90.471 ms<br>1.354 s<br>4.363 s | 88.747 ms<br>1.356 s<br>4.363 s | 102.66 MiB<br>594.26 MiB<br>1.94 GiB | 634270<br>25421514<br>109130299 |
| | | | | | |
| `profile` (one variable) | Regression<br>Linear exponential ODE<br>Logistic ODE | 20.178 ms<br>595.322 ms<br>927.881 ms | 21.788 ms<br>601.882 ms<br>935.553 ms | 25.13 MiB<br>255.62 MiB<br>367.48 MiB | 150089<br>10984510<br>20156577 |
| `profile` (one variable, lowered) | Regression<br>Linear exponential ODE<br>Logistic ODE | 20.314 ms<br>729.048 ms<br>812.594 ms | 21.859 ms<br>775.513 ms<br>813.546 ms | 25.13 MiB<br>255.62 MiB<br>367.48 MiB | 150089<br>10984510<br>20156577 |
| | | | | | |
| `prepare_profile` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.967 μs<br>2.810 μs<br>3.544 μs | 4.442 μs<br>3.312 μs<br>4.244 μs | 6.23 KiB<br>5.09 KiB<br>5.64 KiB | 80<br>50<br>65 |
| `update_prob` (remove bounds) | Regression<br>Linear exponential ODE<br>Logistic ODE | 1.880 μs<br>1.132 μs<br>1.480 μs | 1.874 μs<br>1.450 μs<br>1.544 μs | 1.50 KiB<br>864 bytes<br>1.16 KiB | 46<br>26<br>36 |
| `update_prob` (set initial guess) | Regression<br>Linear exponential ODE<br>Logistic ODE | 5.900 ns<br>3.300 ns<br>4.000 ns | 6.121 ns<br>3.439 ns<br>4.347 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 24.373 ns<br>14.715 ns<br>17.017 ns | 25.162 ns<br>15.943 ns<br>18.451 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `construct_new_f` | Regression<br>Linear exponential ODE<br>Logistic ODE | 17.800 ns<br>11.700 ns<br>13.800 ns | 18.099 ns<br>12.232 ns<br>14.257 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set initial guess and new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 32.596 ns<br>23.323 ns<br>24.048 ns | 34.691 ns<br>23.295 ns<br>25.517 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 31.000 μs<br>12.258 ms<br>20.492 ms | 34.499 μs<br>12.549 ms<br>22.193 ms | 37.66 KiB<br>5.07 MiB<br>9.38 MiB | 242<br>217985<br>513943 |
| `step_profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 505.500 μs<br>11.746 ms<br>11.661 ms | 558.376 μs<br>12.291 ms<br>12.449 ms | 623.58 KiB<br>4.51 MiB<br>3.98 MiB | 3469<br>194093<br>194175 |
| `find_endpoint!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 10.276 ms<br>297.600 ms<br>398.085 ms | 11.109 ms<br>300.707 ms<br>398.571 ms | 12.72 MiB<br>121.80 MiB<br>180.17 MiB | 75905<br>5234607<br>9937392 |
| | | | | | |
| `confidence_intervals` (all) | Regression<br>Linear exponential ODE<br>Logistic ODE | 26.100 μs<br>18.200 μs<br>29.000 μs | 29.142 μs<br>20.628 μs<br>32.371 μs | 41.41 KiB<br>26.67 KiB<br>45.34 KiB | 70<br>44<br>57 |
| `confidence_intervals` (single) | Regression<br>Linear exponential ODE<br>Logistic ODE | 4.671 μs<br>6.940 μs<br>5.229 μs | 5.313 μs<br>7.863 μs<br>5.946 μs | 7.39 KiB<br>11.06 KiB<br>8.22 KiB | 13<br>13<br>13 |
| | | | | | |
| `ProfileLikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 26.406 ns<br>18.700 ns<br>20.762 ns | 27.517 ns<br>20.202 ns<br>21.793 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
