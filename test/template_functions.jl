using LaTeXStrings
using Random
using Distributions
using OptimizationNLopt
using DifferentialEquations
using Test
using PreallocationTools
using LinearAlgebra
using Optimization
using Dierckx
using LoopVectorization

function MultipleLinearRegression()
    Random.seed!(98871)
    n = 300
    β = [-1.0, 1.0, 0.5, 3.0]
    σ = 0.05
    θ₀ = ones(5)
    x₁ = rand(Uniform(-1, 1), n)
    x₂ = rand(Normal(1.0, 0.5), n)
    X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
    ε = rand(Normal(0.0, σ), n)
    y = X * β + ε
    sse = dualcache(zeros(n))
    β_cache = dualcache(similar(β), 10)
    dat = (y, X, sse, n, β_cache)
    @inline function loglik(θ, data)
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
        @turbo for i in eachindex(y)
            ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - sse[i])^2
        end
        return ℓℓ
    end
    prob = LikelihoodProblem(loglik, 5;
        θ₀,
        data=dat,
        adtype=Optimization.AutoForwardDiff(),
        lb=[0.0, -Inf, -Inf, -Inf, -Inf],
        ub=Inf * ones(5),
        names=[L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"])
    return prob, loglik, [σ, β], dat
end

function LinearExponentialODE()
    Random.seed!(2992999)
    λ = -0.5
    y₀ = 15.0
    σ = 0.1
    T = 5.0
    n = 200
    Δt = T / n
    t = [j * Δt for j in 0:n]
    y = y₀ * exp.(λ * t)
    yᵒ = y .+ [0.0, rand(Normal(0, σ), n)...]
    function ode_fnc(u, p, t)
        λ = p
        du = λ * u
        return du
    end
    function loglik(θ, data, integrator)
        ## Extract the parameters
        yᵒ, n = data
        λ, σ, u0 = θ
        ## What do you want to do with the integrator?
        integrator.p = λ
        ## Now solve the problem 
        reinit!(integrator, u0)
        solve!(integrator) # DON'T DO SOL = ... IT CAN CAUSE MEMORY ISSUES OR CHANGE THE REF FOR FUTURE REFERENCES TO VARIABLES CALLED "sol"
        ## Now what do you want to do with sol?
        return gaussian_loglikelihood(yᵒ, integrator.sol.u, σ, n)
    end
    θ₀ = [-1.0, 0.5, 19.73]
    prob = LikelihoodProblem(loglik, 3, ode_fnc, y₀, (0.0, T), 1.0, t;
        data=(yᵒ, n), θ₀, lb=[-10.0, 1e-6, 0.5], ub=[10.0, 10.0, 25.0], ode_kwargs=(verbose=false,),
        names=[L"\lambda", L"\sigma", L"y_0"])
    return prob, loglik, (-0.5, 0.1, 15.0), yᵒ, n
end

function LogisticODE()
    Random.seed!(2929911002)
    u₀ = 0.5
    λ = 1.0
    K = 1.0
    n = 100
    T = 10.0
    t = LinRange(0, T, n)
    u = @. K * u₀ * exp(λ * t) / (K - u₀ + u₀ * exp(λ * t))
    σ = 0.1
    uᵒ = u .+ [0.0, σ * randn(length(u) - 1)...]
    function ode_fnc(u, p, t)
        λ, K = p
        du = λ * u * (1 - u / K)
        return du
    end
    function loglik(θ, data, integrator)
        ## Extract the parameters
        uᵒ, n = data
        λ, K, σ, u0 = θ
        ## What do you want to do with the integrator?
        integrator.p[1] = λ
        integrator.p[2] = K
        ## Now solve the problem 
        reinit!(integrator, u0)
        solve!(integrator)
        return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, n)
    end
    θ₀ = [0.7, 2.0, 0.15, 0.4]
    lb = [0.0, 1e-6, 1e-6, 0.0]
    ub = [10.0, 10.0, 10.0, 10.0]
    param_names = [L"\lambda", L"K", L"\sigma", L"u_0"]
    prob = LikelihoodProblem(loglik, 4, ode_fnc, u₀, (0.0, T), [1.0, 1.0], t;
        data=(uᵒ, n), θ₀, lb, ub, ode_kwargs=(verbose=false,),
        names=param_names, syms=[:λ, :K, :σ, :u₀])
    return prob, loglik, (1.0, 1.0, 0.1, 0.5), uᵒ, n
end