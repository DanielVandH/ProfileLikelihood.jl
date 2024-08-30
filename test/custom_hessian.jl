using Optimization
using ProfileLikelihood
using OptimizationOptimJL
using ForwardDiff
using Test
using SymbolicIndexingInterface

rosenbrock(x, p) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
_∇f = (G, u, p) -> begin
    p = Float64[]
    _f = (x) -> rosenbrock(x, p)
    ForwardDiff.gradient!(G, _f, u)

end
_Δf = (H, u, p) -> begin
    p = Float64[]
    _f = (x) -> rosenbrock(x, p)
    ForwardDiff.hessian!(H, _f, u)
end

x0 = zeros(2) .+ 0.5
xnames = [:x1, :x2]
optimization_function = Optimization.OptimizationFunction(rosenbrock;
    grad=_∇f,
    hess=_Δf,
    sys=SymbolCache([:x1, :x2]; defaults=Dict(:x1 => 0.5, :x2 => 0.5)))
prob = OptimizationProblem(optimization_function, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
_likelihood_problem = LikelihoodProblem{length(x0),
    typeof(prob),
    typeof(SciMLBase.NullParameters()),
    typeof(prob.f),
    typeof(x0),
    typeof(ProfileLikelihood._to_symbolcache(xnames, x0))}(prob,
    SciMLBase.NullParameters(),
    prob.f,
    x0,
    ProfileLikelihood._to_symbolcache(xnames, x0))
sol = mle(_likelihood_problem, Optim.LBFGS())
prof1 = profile(_likelihood_problem, sol)

# Trusting the AD
x0 = zeros(2) .+ 0.5
optimization_function = Optimization.OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff();
    syms=xnames)
prob = OptimizationProblem(optimization_function, x0, lb=[-1.0, -1.0], ub=[0.8, 0.8])
_likelihood_problem = LikelihoodProblem{length(x0),
    typeof(prob),
    typeof(SciMLBase.NullParameters()),
    typeof(prob.f),
    typeof(x0),
    typeof(ProfileLikelihood._to_symbolcache(xnames, x0))}(prob,
    SciMLBase.NullParameters(),
    prob.f,
    x0,
    ProfileLikelihood._to_symbolcache(xnames, x0))
    sol = mle(_likelihood_problem, Optim.LBFGS())
prof2 = profile(_likelihood_problem, sol)

@test prof1.confidence_intervals == prof2.confidence_intervals