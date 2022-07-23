prob1, loglikk, (λ, σ, y₀), yᵒ, n = LinearExponentialODEAutoDiff()
prob2, loglikk, (λ, σ, y₀), yᵒ, n = LinearExponentialODE()
using BenchmarkTools, OptimizationOptimJL
b1 = @benchmark mle(prob1, NLopt.LD_LBFGS; maxtime = 10)
b2 = @benchmark mle(prob1, NLopt.LD_MMA; maxtime = 10)
b3 = @benchmark mle(prob1, NLopt.LN_NELDERMEAD; maxtime = 10)

b4 = @benchmark mle(prob2, NLopt.LD_LBFGS; maxtime = 10)
b5 = @benchmark mle(prob2, NLopt.LD_MMA; maxtime = 10)
b6 = @benchmark mle(prob2, NLopt.LN_NELDERMEAD; maxtime = 10)

a1 = @benchmark mle(prob1, Optim.LBFGS(); maxtime = 10)
a2 = @benchmark mle(prob2, Optim.LBFGS(); maxtime = 10)