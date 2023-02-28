######################################################
## Templates 
######################################################
using StableRNGs
function multiple_linear_regression()
    rng = StableRNG(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.05
    θ₀ = ones(5)
    x₁ = rand(rng, Uniform(-1, 1), n)
    x₂ = rand(rng, Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(rng, Normal(0.0, σ), n)
    y = X * β + ε
    sse = DiffCache(zeros(n))
    β_cache = DiffCache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik(θ, data)
        local σ, y, X, sse, n, β # type stability
        σ, β₀, β₁, β₂, β₃ = θ
        y, X, sse, n, β = data
        sse = get_tmp(sse, θ)
        β = get_tmp(β, θ)
        β[1] = β₀
        β[2] = β₁
        β[3] = β₂
        β[4] = β₃
        ℓℓ = -0.5n * log(2π * σ^2)
        mul!(sse, X, β)
        for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - sse[i])^2
        end
        return ℓℓ
    end
    prob = LikelihoodProblem(loglik, θ₀;
        data=dat,
        f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
        prob_kwargs=(lb=[0.0, -Inf, -Inf, -Inf, -Inf],
            ub=Inf * ones(5)))
    @inferred loglik(prob.θ₀, ProfileLikelihood.get_data(prob))
    @inferred prob.problem.f(prob.θ₀, ProfileLikelihood.get_data(prob))
    return prob, loglik, [σ, β], dat
end