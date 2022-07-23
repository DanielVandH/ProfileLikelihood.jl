using Distributions
using LinearAlgebra
using Random
using LoopVectorization
Random.seed!(2881888211)

μ = [27.0, 31.71, 34.0, 99.0, 1.0]
σ = [5.5, 7.71, 8.3, 18.0, 2.777]
n = [28100, 20070, 58700, 2000070, 1700]
true_dist1 = Normal(μ[1], σ[1])
true_dist2 = Normal(μ[2], σ[2])
true_dist3 = Normal(μ[3], σ[3])
true_dist4 = Normal(μ[4], σ[4])
true_dist5 = Normal(μ[5], σ[5])
x₁ = rand(true_dist1, n[1])
x₂ = rand(true_dist2, n[2])
x₃ = rand(true_dist3, n[3])
x₄ = rand(true_dist4, n[4])
x₅ = rand(true_dist5, n[5])

function loglik(θ, p)
    x₁, x₂, x₃, x₄, x₅ = p
    ℓ = zero(eltype(θ))
    newdist1 = Normal(θ[1], θ[2])
    newdist2 = Normal(θ[3], θ[4])
    newdist3 = Normal(θ[5], θ[6])
    newdist4 = Normal(θ[7], θ[8])
    newdist5 = Normal(θ[9], θ[10])
    @tturbo for i in eachindex(x₁)
        ℓ += logpdf(newdist1, x₁[i])
    end
    @tturbo for i in eachindex(x₂)
        ℓ += logpdf(newdist2, x₂[i])
    end
    @tturbo for i in eachindex(x₃)
        ℓ += logpdf(newdist3, x₃[i])
    end
    @tturbo for i in eachindex(x₄)
        ℓ += logpdf(newdist4, x₄[i])
    end
    @tturbo for i in eachindex(x₅)
        ℓ += logpdf(newdist5, x₅[i])
    end
    ℓ/1.5e6
end

true_θ = [μ[1], σ[1], μ[2], σ[2], μ[3], σ[3], μ[4], σ[4], μ[5], σ[5]]

prob = LikelihoodProblem(
    loglik,
    10;
    θ₀=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    data=(x₁, x₂, x₃, x₄, x₅),
    lb=[-100.0, 1e-12, -100.0, 1e-12, -100.0, 1e-12, -100.0, 1e-12, -100.0, 1e-12],
    ub=[100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0, 100.0, 200.0],
    adtype=Optimization.AutoFiniteDiff()
)
sol = mle(prob, NLopt.LD_LBFGS)
@test norm(mle(sol) - true_θ)/norm(true_θ) < 1e-2
@test abs(maximum(sol) - loglik(true_θ, data(sol)))/abs(maximum(sol)) < 1e-5

prof = profile(prob, sol; alg = NLopt.LD_LBFGS, normalise=true, maxtime = 10, abstol = 1e-4)
fig = plot_profiles(prof;true_vals=true_θ)