using Random, Distributions, StableRNGs, ReferenceTests
rng = StableRNG(98871)
n = 600
β = [-1.0, 1.0, 0.5, 3.0, 1.0, 1.0]
σ = 0.05
x₁ = rand(rng, Normal(0, 0.2), n)
x₂ = rand(rng, Uniform(-1, 1), n)
x₃ = rand(rng, Normal(0, 1), n)
x₄ = rand(rng, Exponential(1), n)
ε = rand(rng, Normal(0, σ), n)
X = hcat(ones(n), x₁, x₂, x₁ .* x₃, x₄)
βcombined = [β[1], β[2], β[3], β[4], β[5] * β[6]]
y = X * βcombined + ε

using PreallocationTools, LinearAlgebra
sse = DiffCache(zeros(n))
βcache = DiffCache(similar(β, length(β) - 1), 10) # -1 because we combine β[5] and β[6]
data = (y, X, sse, n, βcache)
function loglik(θ, data)
    σ, β₀, β₁, β₂, β₃, β₄, β₅ = θ
    β₄β₅ = β₄ * β₅
    y, X, sse, n, β = data
    _sse = get_tmp(sse, θ)
    _β = get_tmp(β, θ)
    _β .= (β₀, β₁, β₂, β₃, β₄β₅)
    ℓℓ = -0.5n * log(2π * σ^2)
    mul!(_sse, X, _β)
    for (yᵢ, sseᵢ) in zip(y, _sse)
        ℓℓ -= 0.5 * (yᵢ - sseᵢ)^2 / σ^2
    end
    return ℓℓ
end

using ..ProfileLikelihood, Optimization
θ₀ = ones(7) # initial guess 
prob = LikelihoodProblem(loglik, θ₀; data,
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(
        lb=[0.0, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
        ub=fill(Inf, 7),
    ),
    syms=[:σ, :β₀, :β₁, :β₂, :β₃, :β₄, :β₅])

using OptimizationNLopt
sol = mle(prob, (NLopt.LN_NELDERMEAD(), NLopt.LD_LBFGS())) # can provide multiple algorithms to run one after the other
prof_lb = [1e-12, -5.0, -5.0, -5.0, -2.0, -5.0, -5.0]
prof_ub = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
resolutions = [1200, 200, 200, 200, 200, 200, 200] # use many points for σ
param_ranges = construct_profile_ranges(sol, prof_lb, prof_ub, resolutions)
prof = profile(prob, sol; param_ranges, parallel=true)

fig = plot_profiles(prof,
    true_vals=[σ, β...],
    axis_kwargs=(width=200, height=200),
    xlim_tuples=[(0.048, 0.056), (-1.01, -0.985), (0.97, 1.050),
        (0.485, 0.505), (2.97, 3.050), (0.95, 1.05),
        (0.95, 1.05)],
    ncol=4, nrow=2
) # see the ?plot_profiles docstring for more options
resize_to_layout!(fig)
fig
@test_reference "figures/profile_likelihood.png" fig

using StaticArrays 
function repar_loglik(θ, data)
    σ, β₀, β₁, β₂, β₃, β₄, β₄β₅ = θ
    θ′ = @SVector[σ, β₀, β₁, β₂, β₃, β₄, β₄β₅/β₄]
    return loglik(θ′, data)
end
prob = LikelihoodProblem(repar_loglik, θ₀; data,
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(
        lb=[0.0, -Inf, -Inf, -Inf, -Inf, 1e-12, -Inf], # 1e-12 to avoid division by zero
        ub=fill(Inf, 7),
    ),
    syms=[:σ, :β₀, :β₁, :β₂, :β₃, :β₄, :β₄β₅])
sol = mle(prob, (NLopt.LN_NELDERMEAD(), NLopt.LD_LBFGS())) 
prof_lb[6] = 1e-12
param_ranges = construct_profile_ranges(sol, prof_lb, prof_ub, resolutions)
prof = profile(prob, sol; param_ranges, parallel=true)
fig = plot_profiles(prof,
    true_vals=[σ, β...],
    axis_kwargs=(width=200, height=200),
    xlim_tuples=[(0.048, 0.056), (-1.01, -0.985), (0.97, 1.050),
        (0.485, 0.505), (2.97, 3.050), (0.95, 1.05),
        (0.99, 1.01)],
    ncol=4, nrow=2
) 
resize_to_layout!(fig)
fig
@test_reference "figures/profile_likelihood_reparam.png" fig