######################################################
## GridSearch
######################################################
## Test that we are constructing the grid correctly 
f = (x, _) -> x[1] * x[2] * x[3]
lb = [2.7, 5.3, 10.0]
ub = [10.0, 7.7, 14.4]
res = 50
ug = PL.RegularGrid(lb, ub, res)
gs = PL.GridSearch(f, ug)
@test gs.f isa FunctionWrappers.FunctionWrapper{Float64,Tuple{Vector{Float64},Nothing}}
@test gs.f.obj[] == f
@test PL.get_grid(gs) == gs.grid == ug
@test PL.get_function(gs) == gs.f
@test PL.eval_function(gs, [1.0, 2.0, 3.0]) ≈ 6.0
@inferred PL.eval_function(gs, rand(3))
@test PL.number_of_parameters(gs) == 3
@test PL.get_parameters(gs) === nothing

f = (x, _) -> x[1] * x[2] * x[3] + x[4]
lb = [2.0, 5.0, 1.3, 5.0]
ub = [5.0, 10.0, 17.3, 20.0]
grid = [rand(2) for _ in 1:200]
ig = PL.IrregularGrid(lb, ub, grid)
gs = PL.GridSearch(f, ig, 2.0)
@test gs.f isa FunctionWrappers.FunctionWrapper{Float64,Tuple{Vector{Float64},Float64}}
@test gs.f.obj[] == f
@test PL.get_grid(gs) == gs.grid == ig
@test PL.get_function(gs) == gs.f
@test PL.eval_function(gs, [1.0, 4.2, 4.2, -1.0]) ≈ f([1.0, 4.2, 4.2, -1.0], 2.0)
@inferred PL.eval_function(gs, rand(4))
@test PL.number_of_parameters(gs) == 4
@test PL.get_parameters(gs) == 2.0

## Test that we are preparing the grid correctly 
lb = [2.0, 5.0, 1.3, 5.0]
ub = [5.0, 10.0, 17.3, 20.0]
grid = [rand(2) for _ in 1:200]
ig = PL.IrregularGrid(lb, ub, grid)
A_ig = PL.prepare_grid(ig)
@test A_ig == zeros(200)

lb = [2.7, 5.3, 10.0]
ub = [10.0, 7.7, 14.4]
res = 50
ug = PL.RegularGrid(lb, ub, res)
A_ug = PL.prepare_grid(ug)
@test A_ug == zeros(50, 50, 50)

lb = [2.7, 5.3, 10.0]
ub = [10.0, 7.7, 14.4]
res = [20, 50, 70]
ug_2 = PL.RegularGrid(lb, ub, res)
A_ug_2 = PL.prepare_grid(ug_2)
@test A_ug_2 == zeros(20, 50, 70)

gs1 = PL.GridSearch(f, ig)
gs2 = PL.GridSearch(f, ug)
gs3 = PL.GridSearch(f, ug_2)
B_ig = PL.prepare_grid(gs1)
B_ug = PL.prepare_grid(gs2)
B_ug_2 = PL.prepare_grid(gs3)
@test B_ig == A_ig
@test B_ug == A_ug
@test B_ug_2 == A_ug_2

## Test that we are getting the correct likelihood function from GridSearch 
loglik = (θ, p) -> -(p[1] - θ[1])^2 - p[2] * (θ[2] - θ[1])^2 + 3.0
θ₀ = zeros(2)
dat = [1.6, 100.0]
prob = LikelihoodProblem(loglik, θ₀; data=dat, syms=[:α, :β],
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),))
lb = [2.7, 5.3, 10.0]
ub = [10.0, 7.7, 14.4]
res = [20, 50, 70]
ug = RegularGrid(lb, ub, res)
gs = GridSearch(prob, ug)
@test PL.eval_function(gs, [1.0, 2.7]) == loglik([1.0, 2.7], dat)
@inferred PL.eval_function(gs, [1.0, 2.7])
@test PL.get_grid(gs) == ug

lb = [2.0, 5.0, 1.3, 5.0]
ub = [5.0, 10.0, 17.3, 20.0]
grid = [rand(2) for _ in 1:200]
ig = IrregularGrid(lb, ub, grid)
gs = GridSearch(prob, ig)
@test PL.eval_function(gs, [1.0, 2.7]) == loglik([1.0, 2.7], dat)
@inferred PL.eval_function(gs, [1.0, 2.7])
@test PL.get_grid(gs) == ig

## Test that the GridSearch works on a set of problems
# Rastrigin function 
n = 4
A = 10
rastrigin_f(x, _) = @inline @inbounds A * n + (x[1]^2 - A * cos(2π * x[1])) + (x[2]^2 - A * cos(2π * x[2])) + (x[3]^2 - A * cos(2π * x[3])) + (x[4]^2 - A * cos(2π * x[4]))
@test rastrigin_f(zeros(n), nothing) == 0.0
ug = PL.RegularGrid(repeat([-5.12], n), repeat([5.12], n), 25)
gs = PL.GridSearch((x, _) -> (@inline; -rastrigin_f(x, nothing)), ug)
f_min, x_min = PL.grid_search(gs)
@inferred PL.grid_search(gs)
@inferred PL.grid_search(gs; save_vals=Val(true))
@inferred PL.grid_search(gs; save_vals=Val(false), parallel=Val(true))
@inferred PL.grid_search(gs; save_vals=Val(true), parallel=Val(true))
@inferred PL.grid_search(gs; save_vals=Val(true), minimise=Val(true))
@inferred PL.grid_search(gs; save_vals=Val(false), parallel=Val(true), minimise=Val(true))
@inferred PL.grid_search(gs; save_vals=Val(true), parallel=Val(true), minimise=Val(true))
@inferred PL.grid_search(gs; minimise=Val(true))

@test f_min ≈ 0.0
@test x_min ≈ zeros(n)

gs = PL.GridSearch(rastrigin_f, ug)
f_min, x_min = grid_search(gs; minimise=Val(true))
@test f_min ≈ 0.0
@test x_min ≈ zeros(n)
@inferred PL.grid_search(gs; minimise=Val(true))

f_min, x_min, f_res = grid_search(gs; minimise=Val(true), save_vals=Val(true))
@test f_min ≈ 0.0
@test x_min ≈ zeros(n)

param_ranges = [LinRange(-5.12, 5.12, 25) for i in 1:n]
@test f_res ≈ [rastrigin_f(x, nothing) for x in Iterators.product(param_ranges...)]

for _ in 1:250
    lb = repeat([-5.12], n)
    ub = repeat([5.12], n)
    gr = rand(n, 2000)
    ig = PL.IrregularGrid(lb, ub, gr)
    f_min, x_min = grid_search(rastrigin_f, ig; minimise=Val(true))
    @inferred grid_search(rastrigin_f, ig; minimise=Val(true))
    @test f_min == minimum(rastrigin_f(x, nothing) for x in eachcol(gr))
    xm = findmin(x -> rastrigin_f(x, nothing), eachcol(gr))[2]
    @test x_min == gr[:, xm]

    gr = [rand(n) for _ in 1:2000]
    ig = PL.IrregularGrid(lb, ub, gr)
    f_min, x_min = grid_search(rastrigin_f, ig; minimise=Val(false))
    @test f_min == maximum(rastrigin_f(x, nothing) for x in gr)
    xm = findmax(x -> rastrigin_f(x, nothing), gr)[2]
    @test x_min == gr[xm]
end

# Ackley function 
n = 2
ackley_f(x, _) = -20exp(-0.2sqrt(x[1]^2 + x[2]^2)) - exp(0.5(cos(2π * x[1]) + cos(2π * x[2]))) + exp(1) + 20
@test ackley_f([0, 0], nothing) == 0
lb = repeat([-15.12], n)
ub = repeat([15.12], n)
res = (73, 121)
ug = RegularGrid(lb, ub, res)
f_min, x_min = grid_search(ackley_f, ug; minimise=Val(true))
@test f_min ≈ 0.0 atol = 1e-7
@test x_min ≈ zeros(n) atol = 1e-7

f_min, x_min = grid_search((x, _) -> -ackley_f(x, nothing), ug; minimise=Val(false))
@test f_min ≈ 0.0 atol = 1e-7
@test x_min ≈ zeros(n) atol = 1e-7

f_min, x_min, f_res = grid_search(ackley_f, ug; minimise=Val(true), save_vals=Val(true))
param_ranges = [LinRange(lb[i], ub[i], res[i]) for i in 1:n]
@test f_res ≈ [ackley_f(x, nothing) for x in Iterators.product(param_ranges...)]

f_min, x_min, f_res = grid_search((x, _) -> -ackley_f(x, nothing), ug; minimise=Val(false), save_vals=Val(true))
@test f_res ≈ [-ackley_f(x, nothing) for x in Iterators.product(param_ranges...)]

for _ in 1:250
    gr = rand(n, 500)
    ig = IrregularGrid(lb, ub, gr)
    f_min, x_min = grid_search(ackley_f, ig; minimise=Val(true))
    @inferred grid_search(ackley_f, ig; minimise=Val(true))
    @test f_min == minimum(ackley_f(x, nothing) for x in eachcol(gr))
    xm = findmin(x -> ackley_f(x, nothing), eachcol(gr))[2]
    @test x_min == gr[:, xm]

    gr = [rand(n) for _ in 1:2000]
    ig = IrregularGrid(lb, ub, gr)
    f_min, x_min = grid_search(ackley_f, ig; minimise=Val(false))
    @inferred grid_search(ackley_f, ig; minimise=Val(true))
    @test f_min == maximum(ackley_f(x, nothing) for x in gr)
    xm = findmax(x -> ackley_f(x, nothing), gr)[2]
    @test x_min == gr[xm]

    gr = rand(n, 500)
    ig = IrregularGrid(lb, ub, gr)
    f_min, x_min = grid_search(ackley_f, ig; minimise=Val(true), parallel=Val(true))
    @inferred grid_search(ackley_f, ig; minimise=Val(true), parallel=Val(true))
    @test f_min == minimum(ackley_f(x, nothing) for x in eachcol(gr))
    xm = findmin(x -> ackley_f(x, nothing), eachcol(gr))[2]
    @test x_min == gr[:, xm]

    gr = [rand(n) for _ in 1:2000]
    ig = IrregularGrid(lb, ub, gr)
    f_min, x_min = grid_search(ackley_f, ig; minimise=Val(false), parallel=Val(true))
    @inferred grid_search(ackley_f, ig; minimise=Val(true), parallel=Val(true))
    @test f_min == maximum(ackley_f(x, nothing) for x in gr)
    xm = findmax(x -> ackley_f(x, nothing), gr)[2]
    @test x_min == gr[xm]
end

# Sphere function 
n = 5
sphere_f(x, _) = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2
@test sphere_f(zeros(n), nothing) == 0
lb = repeat([-15.12], n)
ub = repeat([15.12], n)
res = (15, 17, 15, 19, 23)
ug = RegularGrid(lb, ub, res)
f_min, x_min = grid_search(sphere_f, ug; minimise=Val(true))
@test f_min ≈ 0.0 atol = 1e-7
@test x_min ≈ zeros(n) atol = 1e-7

f_min, x_min = grid_search((x, _) -> -sphere_f(x, nothing), ug; minimise=Val(false))
@test f_min ≈ 0.0 atol = 1e-7
@test x_min ≈ zeros(n) atol = 1e-7

f_min, x_min, f_res = grid_search(sphere_f, ug; minimise=Val(true), save_vals=Val(true))
param_ranges = [LinRange(lb[i], ub[i], res[i]) for i in 1:n]
@test f_res ≈ [sphere_f(x, nothing) for x in Iterators.product(param_ranges...)]
f_min, x_min, f_res = grid_search((x, _) -> -sphere_f(x, nothing), ug; minimise=Val(false), save_vals=Val(true))
@test f_res ≈ [-sphere_f(x, nothing) for x in Iterators.product(param_ranges...)]

f_min, x_min, f_res = grid_search(sphere_f, ug; minimise=Val(true), save_vals=Val(true), parallel=Val(true))
param_ranges = [LinRange(lb[i], ub[i], res[i]) for i in 1:n]
@test f_res ≈ [sphere_f(x, nothing) for x in Iterators.product(param_ranges...)]
f_min, x_min, f_res = grid_search((x, _) -> -sphere_f(x, nothing), ug; minimise=Val(false), save_vals=Val(true), parallel=Val(true))
@test f_res ≈ [-sphere_f(x, nothing) for x in Iterators.product(param_ranges...)]

for _ in 1:250
    gr = rand(n, 500)
    ig = IrregularGrid(lb, ub, gr)
    f_min, x_min = grid_search(sphere_f, ig; minimise=Val(true))
    @inferred grid_search(sphere_f, ig; minimise=Val(true))
    @test f_min == minimum(sphere_f(x, nothing) for x in eachcol(gr))
    xm = findmin(x -> sphere_f(x, nothing), eachcol(gr))[2]
    @test x_min == gr[:, xm]

    gr = rand(n, 500)
    ig = IrregularGrid(lb, ub, gr)
    f_min, x_min, f_res = grid_search(sphere_f, ig; minimise=Val(true), save_vals=Val(true))
    @inferred grid_search(sphere_f, ig; minimise=Val(true), save_vals=Val(true))
    @test f_min == minimum(sphere_f(x, nothing) for x in eachcol(gr))
    xm = findmin(x -> sphere_f(x, nothing), eachcol(gr))[2]
    @test x_min == gr[:, xm]
    @test f_res ≈ [(sphere_f(x, nothing) for x in eachcol(gr))...]

    gr = [rand(n) for _ in 1:2000]
    ig = IrregularGrid(lb, ub, gr)
    f_min, x_min = grid_search(sphere_f, ig; minimise=Val(false))
    @test f_min == maximum(sphere_f(x, nothing) for x in gr)
    xm = findmax(x -> sphere_f(x, nothing), gr)[2]
    @test x_min == gr[xm]

    gr = [rand(n) for _ in 1:2000]
    ig = IrregularGrid(lb, ub, gr)
    f_min, x_min, f_res = grid_search(sphere_f, ig; minimise=Val(false), save_vals=Val(true))
    @test f_min == maximum(sphere_f(x, nothing) for x in gr)
    xm = findmax(x -> sphere_f(x, nothing), gr)[2]
    @test x_min == gr[xm]
    @test f_res ≈ [(sphere_f(x, nothing) for x in gr)...]
end

# Multiple Linear Regression
prob, loglikk, θ, dat = multiple_linear_regression()
true_ℓ = loglikk(reduce(vcat, θ), dat)
lb = (1e-12, -3.0, 0.0, 0.0, -6.0)
ub = (0.2, 0.0, 3.0, 3.0, 6.0)
res = 27
ug = RegularGrid(lb, ub, res)
sol = grid_search(prob, ug; save_vals=Val(false))
@inferred grid_search(prob, ug; save_vals=Val(false))
@test sol isa PL.LikelihoodSolution
@test PL.get_maximum(sol) ≈ 281.7360323629172
@test PL.get_mle(sol) ≈ [0.09230769230823077
    -0.9230769230769231
    0.8076923076923077
    0.46153846153846156
    3.2307692307692317]
param_ranges = [LinRange(lb[i], ub[i], res) for i in eachindex(lb)]
f_res_true = [loglikk(collect(x), dat) for x in Iterators.product(param_ranges...)]
@test PL.get_maximum(sol) ≈ maximum(f_res_true)
max_idx = Tuple(findmax(f_res_true)[2])
@test PL.get_mle(sol) ≈ [ug[i, max_idx[i]] for i in eachindex(lb)]

sol, f_res = grid_search(prob, ug; save_vals=Val(true))
@inferred grid_search(prob, ug; save_vals=Val(true))
@test f_res ≈ f_res_true
@test PL.get_maximum(sol) ≈ 281.7360323629172
@test PL.get_mle(sol) ≈ [0.09230769230823077
    -0.9230769230769231
    0.8076923076923077
    0.46153846153846156
    3.2307692307692317]

Random.seed!(82882828)
gr = Matrix(reduce(hcat, [lb[i] .+ (ub[i] - lb[i]) .* rand(250) for i in eachindex(lb)])')
gr = hcat(gr, [0.09230769230823077, -0.9230769230769231, 0.8076923076923077, 0.46153846153846156, 3.2307692307692317])
ig = IrregularGrid(lb, ub, gr)
sol = grid_search(prob, ig; save_vals=Val(false))
@inferred grid_search(prob, ig; save_vals=Val(false))
@test sol isa PL.LikelihoodSolution
@test PL.get_maximum(sol) ≈ 281.7360323629172
@test PL.get_mle(sol) ≈ [0.09230769230823077
    -0.9230769230769231
    0.8076923076923077
    0.46153846153846156
    3.2307692307692317]
f_res_true = [loglikk(x, dat) for x in eachcol(gr)]
@test PL.get_maximum(sol) ≈ maximum(f_res_true)
max_idx = findmax(f_res_true)[2]
@test PL.get_mle(sol) ≈ gr[:, max_idx] == PL.get_parameters(ig, max_idx)

sol, f_res = grid_search(prob, ig; save_vals=Val(true))
@inferred grid_search(prob, ig; save_vals=Val(true))
@test f_res ≈ f_res_true
@test PL.get_maximum(sol) ≈ 281.7360323629172
@test PL.get_mle(sol) ≈ [0.09230769230823077
    -0.9230769230769231
    0.8076923076923077
    0.46153846153846156
    3.2307692307692317]

Random.seed!(82882828)
gr = Matrix(reduce(hcat, [lb[i] .+ (ub[i] - lb[i]) .* rand(250) for i in eachindex(lb)])')
gr = hcat(gr, [0.09230769230823077, -0.9230769230769231, 0.8076923076923077, 0.46153846153846156, 3.2307692307692317])
gr = [collect(x) for x in eachcol(gr)]
ig = IrregularGrid(lb, ub, gr)
sol = grid_search(prob, ig; save_vals=Val(false))
@inferred grid_search(prob, ig; save_vals=Val(false))
@test sol isa PL.LikelihoodSolution
@test PL.get_maximum(sol) ≈ 281.7360323629172
@test PL.get_mle(sol) ≈ [0.09230769230823077
    -0.9230769230769231
    0.8076923076923077
    0.46153846153846156
    3.2307692307692317]
f_res_true = [loglikk(x, dat) for x in gr]
@test PL.get_maximum(sol) ≈ maximum(f_res_true)
max_idx = findmax(f_res_true)[2]
@test PL.get_mle(sol) ≈ gr[max_idx] == PL.get_parameters(ig, max_idx)

sol, f_res = grid_search(prob, ig; save_vals=Val(true))
@inferred grid_search(prob, ig; save_vals=Val(true))
@test f_res ≈ f_res_true
@test PL.get_maximum(sol) ≈ 281.7360323629172
@test PL.get_mle(sol) ≈ [0.09230769230823077
    -0.9230769230769231
    0.8076923076923077
    0.46153846153846156
    3.2307692307692317]

## Test that the parallel grid searches are working 
# Rastrigin
n = 4
A = 10
rastrigin_f(x, _) = @inline @inbounds 10 * 4 + (x[1]^2 - 10 * cos(2π * x[1])) + (x[2]^2 - 10 * cos(2π * x[2])) + (x[3]^2 - 10 * cos(2π * x[3])) + (x[4]^2 - 10 * cos(2π * x[4]))
@test rastrigin_f(zeros(n), nothing) == 0.0
ug = RegularGrid(repeat([-5.12], n), repeat([5.12], n), 45)
gs = GridSearch(rastrigin_f, ug)
f_op_par, x_ar_par, f_res_par = grid_search(gs; minimise=Val(true), save_vals=Val(true), parallel=Val(true))
f_op_ser, x_ar_ser, f_res_ser = grid_search(gs; minimise=Val(true), save_vals=Val(true))
@test f_res_ser == f_res_par
@test f_op_par == f_op_ser
@test x_ar_par == x_ar_ser

# Multiple Linear Regression
prob, loglikk, θ, dat = multiple_linear_regression()
true_ℓ = loglikk(reduce(vcat, θ), dat)
lb = (1e-12, -3.0, 0.0, 0.0, -6.0)
ub = (0.2, 0.0, 3.0, 3.0, 6.0)
res = 27
ug = RegularGrid(lb, ub, res)
sol, L1 = grid_search(prob, ug; save_vals=Val(true))
sol2, L2 = grid_search(prob, ug; save_vals=Val(true), parallel=Val(true))
@test L1 ≈ L2
@test PL.get_mle(sol) ≈ PL.get_mle(sol2)

f = prob.log_likelihood_function
p = prob.data
gs = GridSearch(f, ug, p)
_opt, _sol, _L1 = grid_search(gs; save_vals=Val(true), minimise=Val(false))
_opt2, _sol2, _L2 = grid_search(gs; save_vals=Val(true), parallel=Val(true), minimise=Val(false))
@test L1 == _L1
@test _opt ≈ sol.maximum
@test _sol ≈ sol.mle
@test L2 ≈ _L2
@test _opt2 ≈ sol2.maximum
@test _sol2 ≈ sol2.mle
@test _L1 ≈ _L2

res = 27
gr = [lb[i] .+ (ub[i] - lb[i]) * rand(20) for i in eachindex(lb)]
gr = Matrix(reduce(hcat, gr)')
ug = IrregularGrid(lb, ub, gr)
sol, L1 = grid_search(prob, ug; save_vals=Val(true))
sol2, L2 = grid_search(prob, ug; save_vals=Val(true), parallel=Val(true))
@test L1 ≈ L2
@test PL.get_mle(sol) ≈ PL.get_mle(sol2)

f = prob.log_likelihood_function
p = prob.data
gs = GridSearch(f, ug, p)
_opt, _sol, _L1 = grid_search(gs; save_vals=Val(true), minimise=Val(false))
_opt2, _sol2, _L2 = grid_search(gs; save_vals=Val(true), parallel=Val(true), minimise=Val(false))
@test L1 == _L1
@test _opt ≈ sol.maximum
@test _sol ≈ sol.mle
@test L2 ≈ _L2
@test _opt2 ≈ sol2.maximum
@test _sol2 ≈ sol2.mle
@test _L1 ≈ _L2

# Logistic 
Random.seed!(2929911002)
u₀, λ, K, n, T = 0.5, 1.0, 1.0, 100, 10.0
t = LinRange(0, T, n)
u = @. K * u₀ * exp(λ * t) / (K - u₀ + u₀ * exp(λ * t))
σ = 0.1
uᵒ = u .+ [0.0, σ * randn(length(u) - 1)...] # add some noise 
@inline function ode_fnc(u, p, t)
    local λ, K
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
@inline function loglik_fnc2(θ::AbstractVector{T}, data, integrator) where {T}
    local uᵒ, n, λ, K, σ, u0
    uᵒ, n = data
    λ, K, σ, u0 = θ
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u0)
    solve!(integrator)
    if !SciMLBase.successful_retcode(integrator.sol)
        return typemin(T)
    end
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, n)
end
θ₀ = [0.7, 2.0, 0.15, 0.4]
lb = [0.0, 1e-6, 1e-6, 0.0]
ub = [10.0, 10.0, 10.0, 10.0]
syms = [:λ, :K, :σ, :u₀]
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc, u₀, (0.0, T); # u₀ is just a placeholder IC in this case
    syms=syms,
    data=(uᵒ, n),
    ode_parameters=[1.0, 1.0], # temp values for [λ, K],
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Tsit5()
)
ug = RegularGrid(get_lower_bounds(prob), get_upper_bounds(prob), [36, 36, 36, 36])
sol, L1 = grid_search(prob, ug; save_vals=Val(true))
sol2, L2 = grid_search(prob, ug; save_vals=Val(true), parallel=Val(true))
@test L1 ≈ L2
@test PL.get_mle(sol) ≈ PL.get_mle(sol2)

#b1 = @benchmark grid_search($prob, $ug; save_vals=$Val(false))
#b2 = @benchmark grid_search($prob, $ug; save_vals=$Val(false), parallel=$Val(true))

@inferred grid_search(prob, ug; save_vals=Val(true))
@inferred grid_search(prob, ug; save_vals=Val(true), parallel=Val(true))

prob_copies = [deepcopy(prob) for _ in 1:Base.Threads.nthreads()]
gs = [GridSearch(prob_copies[i].log_likelihood_function, ug, prob_copies[i].data) for i in 1:Base.Threads.nthreads()]
_opt, _sol, _L1 = grid_search(gs[1]; save_vals=Val(true), minimise=Val(false))
_opt2, _sol2, _L2 = grid_search(gs; save_vals=Val(true), parallel=Val(true), minimise=Val(false))
@test L1 == _L1
@test _opt ≈ sol.maximum
@test _sol ≈ sol.mle
@test L2 ≈ _L2
@test _opt2 ≈ sol2.maximum
@test _sol2 ≈ sol2.mle
@test _L1 ≈ _L2

res = 27
gr = [lb[i] .+ (ub[i] - lb[i]) * rand(20) for i in eachindex(lb)]
gr = Matrix(reduce(hcat, gr)')
ug = IrregularGrid(lb, ub, gr)
sol, L1 = grid_search(prob, ug; save_vals=Val(true))
sol2, L2 = grid_search(prob, ug; save_vals=Val(true), parallel=Val(true))
@test L1 ≈ L2
@test PL.get_mle(sol) ≈ PL.get_mle(sol2)

prob_copies = [deepcopy(prob) for _ in 1:Base.Threads.nthreads()]
gs = [GridSearch(prob_copies[i].log_likelihood_function, ug, prob_copies[i].data) for i in 1:Base.Threads.nthreads()]
_opt, _sol, _L1 = grid_search(gs[1]; save_vals=Val(true), minimise=Val(false))
_opt2, _sol2, _L2 = grid_search(gs; save_vals=Val(true), parallel=Val(true), minimise=Val(false))
@test L1 == _L1
@test _opt ≈ sol.maximum
@test _sol ≈ sol.mle
@test L2 ≈ _L2
@test _opt2 ≈ sol2.maximum
@test _sol2 ≈ sol2.mle
@test _L1 ≈ _L2