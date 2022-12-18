######################################################
## Example II: Logistic ODE
######################################################
## Step 1: Generate the data and define the likelihood
λ = 0.01
K = 100.0
u₀ = 10.0
t = 0:100:1000
σ = 10.0
@inline function ode_fnc(u, p, t)
    λ, K = p
    du = λ * u * (1 - u / K)
    return du
end
# Initial data is obtained by solving the ODE 
tspan = extrema(t)
p = (λ, K)
prob = ODEProblem(ode_fnc, u₀, tspan, p)
sol = solve(prob, Rosenbrock23(), saveat=t)
Random.seed!(2828)
uᵒ = sol.u + σ * randn(length(t))
@inline function loglik_fnc2(θ, data, integrator)
    λ, K, u₀ = θ
    uᵒ, σ = data
    integrator.p[1] = λ
    integrator.p[2] = K
    reinit!(integrator, u₀)
    solve!(integrator)
    return gaussian_loglikelihood(uᵒ, integrator.sol.u, σ, length(uᵒ))
end

## Step 2: Define the problem 
lb = [0.0, 50.0, 0.0] # λ, K, u₀
ub = [0.05, 150.0, 50.0]
θ₀ = [λ, K, u₀]
syms = [:λ, :K, :u₀]
prob = LikelihoodProblem(
    loglik_fnc2, θ₀, ode_fnc, u₀, maximum(t); # Note that u₀ is just a placeholder IC in this case since we are estimating it
    syms=syms,
    data=(uᵒ, σ),
    ode_parameters=[1.0, 1.0], # temp values for [λ, K]
    ode_kwargs=(verbose=false, saveat=t),
    f_kwargs=(adtype=Optimization.AutoFiniteDiff(),),
    prob_kwargs=(lb=lb, ub=ub),
    ode_alg=Rosenbrock23()
)

## Step 3: Compute the MLE 
sol = mle(prob, NLopt.LD_LBFGS)
@test get_maximum(sol) ≈ -38.99053694428977 rtol = 1e-3
@test get_mle(sol, 1) ≈ 0.010438031266786045 rtol = 1e-3
@test get_mle(sol, 2) ≈ 99.59921873132551 rtol = 1e-3
@test sol[:u₀] ≈ 8.098422110755225 rtol = 1e-3

## Step 4: Profile 
_prob = deepcopy(prob)
_sol = deepcopy(sol)
prof = profile(prob, sol;
    alg=NLopt.LN_NELDERMEAD, parallel=false)
@test sol.mle == _sol.mle
@test sol.maximum == _sol.maximum # checking aliasing 
@test _prob.θ₀ == prob.θ₀
@test λ ∈ get_confidence_intervals(prof, :λ)
@test K ∈ prof.confidence_intervals[2]
@test u₀ ∈ get_confidence_intervals(prof, 3)


prof1 = profile(prob, sol; parallel=false)
prof2 = profile(prob, sol; parallel=true)

#b1 = @benchmark profile($prob, $sol;  parallel=$true)
#b2 = @benchmark profile($prob, $sol; parallel=$false)


@test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-3
@test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-3
@test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-3
@test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-2
@test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-3
@test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-3
@test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-2
@test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-2
@test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-2
@test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-3
@test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
@test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
@test issorted(prof1.parameter_values[1])
@test issorted(prof1.parameter_values[2])
@test issorted(prof1.parameter_values[3])
@test issorted(prof2.parameter_values[1])
@test issorted(prof2.parameter_values[2])
@test issorted(prof2.parameter_values[3])
@test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
@test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
@test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
@test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
@test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
@test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots

prof = prof1

using CairoMakie, LaTeXStrings
fig = plot_profiles(prof;
    latex_names=[L"\lambda", L"K", L"u_0"],
    show_mles=true,
    shade_ci=true,
    nrow=1,
    ncol=3,
    true_vals=[λ, K, u₀],
    fig_kwargs=(fontsize=30, resolution=(2109.644f0, 444.242f0)),
    axis_kwargs=(width=600, height=300))
SAVE_FIGURE && save("figures/logistic_example.png", fig)

## Step 5: Get prediction intervals, compare to evaluating at many points 
function prediction_function(θ, data)
    λ, K, u₀ = θ
    t = data
    prob = ODEProblem(ode_fnc, u₀, extrema(t), (λ, K))
    sol = solve(prob, Rosenbrock23(), saveat=t)
    return sol.u
end
t_many_pts = LinRange(extrema(t)..., 1000)
parameter_wise, union_intervals, all_curves, param_range = get_prediction_intervals(prediction_function, prof, t_many_pts; q_type=Vector{Float64})

# Get the exact solution and MLE solutions first for comparison 
exact_soln = prediction_function([λ, K, u₀], t_many_pts)
mle_soln = prediction_function(get_mle(sol), t_many_pts)

# Now plot the prediction intervals
fig = Figure(fontsize=38, resolution=(1402.7681f0, 848.64404f0))
alp = join('a':'z')
latex_names = [L"\lambda", L"K", L"u_0"]
for i in 1:3
    ax = Axis(fig[i < 3 ? 1 : 2, i < 3 ? i : 1], title=L"(%$(alp[i])): Profile-wise PI for %$(latex_names[i])",
        titlealign=:left, width=600, height=300)
    [lines!(ax, t_many_pts, all_curves[i][j], color=:grey) for j in eachindex(param_range[1])]
    lines!(ax, t_many_pts, exact_soln, color=:red)
    lines!(ax, t_many_pts, mle_soln, color=:blue, linestyle=:dash)
    lines!(ax, t_many_pts, getindex.(parameter_wise[i], 1), color=:black, linewidth=3)
    lines!(ax, t_many_pts, getindex.(parameter_wise[i], 2), color=:black, linewidth=3)
end
ax = Axis(fig[2, 2], title=L"(d):$ $ Union of all intervals",
    titlealign=:left, width=600, height=300)
band!(ax, t_many_pts, getindex.(union_intervals, 1), getindex.(union_intervals, 2), color=:grey)
lines!(ax, t_many_pts, getindex.(union_intervals, 1), color=:black, linewidth=3)
lines!(ax, t_many_pts, getindex.(union_intervals, 2), color=:black, linewidth=3)
lines!(ax, t_many_pts, exact_soln, color=:red)
lines!(ax, t_many_pts, mle_soln, color=:blue, linestyle=:dash)

# We can now compare these intervals to the one obtained from the full likelihood. We re-use our grid_search code (see the next example) to evaluate at many points 
lb = get_lower_bounds(prob)
ub = get_upper_bounds(prob)
N = 1e5
grid = [[lb[i] + (ub[i] - lb[i]) * rand() for _ in 1:N] for i in 1:3]
grid = permutedims(reduce(hcat, grid), (2, 1))
ig = IrregularGrid(lb, ub, grid)
gs, lik_vals = grid_search(prob, ig; parallel=Val(true), save_vals=Val(true))
lik_vals .-= get_maximum(sol) # normalised 
feasible_idx = findall(lik_vals .> ProfileLikelihood.get_chisq_threshold(0.95)) # values in the confidence region 
parameter_evals = grid[:, feasible_idx]
q = [prediction_function(θ, t_many_pts) for θ in eachcol(parameter_evals)]
q_mat = reduce(hcat, q)
q_lwr = minimum(q_mat; dims=2) |> vec
q_upr = maximum(q_mat; dims=2) |> vec
lines!(ax, t_many_pts, q_lwr, color=:magenta, linewidth=3)
lines!(ax, t_many_pts, q_upr, color=:magenta, linewidth=3)

SAVE_FIGURE && save("figures/logistic_example_prediction.png", fig)