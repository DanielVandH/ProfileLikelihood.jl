using Optimization
using ProfileLikelihood
using Ipopt
using OptimizationMOI
using ForwardDiff
using Test


rosenbrock(x, p) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
_∇f = (G, u, p) ->  begin 
    p = Float64[]
    _f = (x) -> rosenbrock(x, p)
    ForwardDiff.gradient!(G, _f, u)

end
_Δf = (H, u, p) ->  begin 
    p = Float64[]
    _f = (x) -> rosenbrock(x, p)
    ForwardDiff.hessian!(H, _f, u)
end

x0 = zeros(2) .+ 0.5
xnames = [:x1, :x2]
optimization_function = Optimization.OptimizationFunction(rosenbrock;
                                                          grad = _∇f,
                                                          hess = _Δf,
                                                          syms=xnames)
prob = OptimizationProblem(optimization_function, x0, lb = [-1.0, -1.0], ub = [0.8, 0.8])
_likelihood_problem = LikelihoodProblem{length(x0),
                                        typeof(prob),
                                        typeof(SciMLBase.NullParameters()),
                                        typeof(prob.f),
                                        typeof(x0),
                                        typeof(xnames)}(prob,
                                                        SciMLBase.NullParameters(),
                                                        prob.f,
                                                        x0,
                                                        xnames)
sol = mle(_likelihood_problem, Ipopt.Optimizer())
prof1 = profile(_likelihood_problem, sol)

# Trusting the AD
x0 = zeros(2) .+ 0.5
optimization_function = Optimization.OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff();
                                                          syms=xnames)
prob = OptimizationProblem(optimization_function, x0, lb = [-1.0, -1.0], ub = [0.8, 0.8])
_likelihood_problem = LikelihoodProblem{length(x0),
                                        typeof(prob),
                                        typeof(SciMLBase.NullParameters()),
                                        typeof(prob.f),
                                        typeof(x0),
                                        typeof(xnames)}(prob,
                                                        SciMLBase.NullParameters(),
                                                        prob.f,
                                                        x0,
                                                        xnames)
sol = mle(_likelihood_problem, Ipopt.Optimizer())
prof2 = profile(_likelihood_problem, sol)

@test prof1.confidence_intervals == prof2.confidence_intervals