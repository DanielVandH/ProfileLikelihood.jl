# Optimiser benchmarks
## BasicsThe table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and $ n = 300$ data points.
   - Linear exponential ODE: This is related to the problem     $    \mathrm dy/\mathrm dt = \lambda y $ with initial condition $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $    \mathrm dy/\mathrm dt = \lambda y (1 - u/K) $ with initial condition $ y(0) = y_0  $. There are four parameters and $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective (Optimization) | Regression<br>Linear exponential ODE<br>Logistic ODE | 268.171 ns<br>66.300 μs<br>58.200 μs | 283.175 ns<br>73.243 μs<br>61.107 μs | 80 bytes<br>31.38 KiB<br>29.81 KiB | 2<br>1431<br>1619 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.521 ms<br>23.079 ms<br>30.279 ms | 4.002 ms<br>23.809 ms<br>30.836 ms | 3.52 MiB<br>9.63 MiB<br>14.56 MiB | 8486<br>417012<br>811132 |
| `Optimization.solve` | Regression<br>Linear exponential ODE<br>Logistic ODE | 2.530 ms<br>22.572 ms<br>30.341 ms | 2.946 ms<br>22.704 ms<br>30.792 ms | 3.52 MiB<br>9.63 MiB<br>14.56 MiB | 8486<br>417012<br>811132 |
| `LikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 16.032 ns<br>8.400 ns<br>8.900 ns | 17.277 ns<br>8.657 ns<br>9.455 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile` (all variables) | Regression<br>Linear exponential ODE<br>Logistic ODE | 92.835 ms<br>1.547 s<br>4.133 s | 93.170 ms<br>1.557 s<br>4.133 s | 102.66 MiB<br>594.26 MiB<br>1.94 GiB | 634270<br>25421514<br>109130299 |
| | | | | | |
| `profile` (one variable) | Regression<br>Linear exponential ODE<br>Logistic ODE | 22.605 ms<br>604.888 ms<br>765.782 ms | 24.737 ms<br>603.514 ms<br>764.678 ms | 25.13 MiB<br>255.62 MiB<br>367.48 MiB | 150089<br>10984510<br>20156577 |
| `profile` (one variable, lowered) | Regression<br>Linear exponential ODE<br>Logistic ODE | 21.187 ms<br>593.629 ms<br>774.344 ms | 22.964 ms<br>596.141 ms<br>775.526 ms | 25.13 MiB<br>255.62 MiB<br>367.48 MiB | 150089<br>10984510<br>20156577 |
| | | | | | |
| `prepare_profile` | Regression<br>Linear exponential ODE<br>Logistic ODE | 3.711 μs<br>2.720 μs<br>3.456 μs | 4.283 μs<br>3.005 μs<br>3.955 μs | 6.23 KiB<br>5.09 KiB<br>5.64 KiB | 80<br>50<br>65 |
| `update_prob` (remove bounds) | Regression<br>Linear exponential ODE<br>Logistic ODE | 1.900 μs<br>838.889 ns<br>1.390 μs | 2.171 μs<br>1.046 μs<br>1.424 μs | 1.50 KiB<br>864 bytes<br>1.16 KiB | 46<br>26<br>36 |
| `update_prob` (set initial guess) | Regression<br>Linear exponential ODE<br>Logistic ODE | 6.100 ns<br>3.200 ns<br>3.900 ns | 6.218 ns<br>3.291 ns<br>4.005 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 24.975 ns<br>14.314 ns<br>17.000 ns | 26.048 ns<br>14.746 ns<br>18.268 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `construct_new_f` | Regression<br>Linear exponential ODE<br>Logistic ODE | 18.236 ns<br>11.411 ns<br>13.413 ns | 19.342 ns<br>11.743 ns<br>13.923 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| `update_prob` (set initial guess and new objective) | Regression<br>Linear exponential ODE<br>Logistic ODE | 33.434 ns<br>19.400 ns<br>23.470 ns | 34.798 ns<br>19.950 ns<br>24.224 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
| | | | | | |
| `profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 32.900 μs<br>11.633 ms<br>19.531 ms | 37.651 μs<br>11.837 ms<br>19.566 ms | 37.66 KiB<br>5.07 MiB<br>9.38 MiB | 242<br>217985<br>513943 |
| `step_profile!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 515.800 μs<br>11.043 ms<br>10.945 ms | 598.013 μs<br>11.299 ms<br>11.306 ms | 623.58 KiB<br>4.51 MiB<br>3.98 MiB | 3469<br>194093<br>194175 |
| `find_endpoint!` | Regression<br>Linear exponential ODE<br>Logistic ODE | 10.863 ms<br>283.389 ms<br>382.954 ms | 12.082 ms<br>287.508 ms<br>382.346 ms | 12.72 MiB<br>121.80 MiB<br>180.17 MiB | 75905<br>5234607<br>9937392 |
| | | | | | |
| `confidence_intervals` (all) | Regression<br>Linear exponential ODE<br>Logistic ODE | 27.900 μs<br>17.700 μs<br>28.700 μs | 32.594 μs<br>19.624 μs<br>32.286 μs | 41.41 KiB<br>26.67 KiB<br>45.34 KiB | 70<br>44<br>57 |
| `confidence_intervals` (single) | Regression<br>Linear exponential ODE<br>Logistic ODE | 4.871 μs<br>6.940 μs<br>5.029 μs | 5.565 μs<br>8.064 μs<br>5.769 μs | 7.39 KiB<br>11.06 KiB<br>8.22 KiB | 13<br>13<br>13 |
| | | | | | |
| `ProfileLikelihoodSolution` | Regression<br>Linear exponential ODE<br>Logistic ODE | 27.309 ns<br>18.200 ns<br>21.000 ns | 28.917 ns<br>18.731 ns<br>21.494 ns | 0 bytes<br>0 bytes<br>0 bytes | 0<br>0<br>0 |
