# Optimiser benchmarks
## Basics
The table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:
   - Regression: This is a multiple linear regression problem. There are five parameters and    $n = 300 $ data points.
   - Linear exponential ODE: This is related to the problem     $\mathrm dy/\mathrm dt = \lambda y $ with initial condition   $ y(0) = y_0   $. There are three parameters and $ n = 200$ data points.
   - Logistic ODE: This is related to the problem     $\mathrm du/\mathrm dt = \lambda u (1 - u/K)$ with initial condition   $u(0) = u_0  $.    There are four parameters and   $ n = 100$ data points.

| Function | Problem | Median Time | Mean Time | Memory | Allocations |
|---|---|--:|--:|--:|--:|
| Objective | Regression<br>Linear exponential ODE<br>Logistic ODE<br>Linear exponential ODE (AutoDiff) | 327.795 ns<br>34.600 μs<br>21.600 μs<br>110.800 μs | 327.795 ns<br>34.600 μs<br>21.600 μs<br>110.800 μs | 80 bytes<br>8.08 KiB<br>2.45 KiB<br>29.22 KiB | 2<br>13<br>12<br>303 |
| | | | | | |
| `mle` | Regression<br>Linear exponential ODE<br>Logistic ODE<br>Linear exponential ODE (AutoDiff) | 2.556 ms<br>8.472 ms<br>6.173 ms<br>12.941 ms | 2.556 ms<br>8.472 ms<br>6.173 ms<br>12.941 ms | 3.49 MiB<br>3.03 MiB<br>1.20 MiB<br>6.56 MiB | 8123<br>7216<br>8602<br>65115 |
| | | | | | |
| `profile` | Regression<br>Linear exponential ODE<br>Logistic ODE<br>Linear exponential ODE (AutoDiff) | 314.043 ms<br>727.043 ms<br>3.341 s<br>1.426 s | 314.043 ms<br>727.043 ms<br>3.341 s<br>1.426 s | 329.65 MiB<br>270.27 MiB<br>887.43 MiB<br>679.07 MiB | 1931188<br>1148348<br>10460349<br>7133205 |
