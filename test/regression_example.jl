######################################################
## Example I: Multiple Linear Regression 
######################################################
## Step 1: Generate some data for the problem and define the likelihood
Random.seed!(98871)
n = 600
β = [-1.0, 1.0, 0.5, 3.0]
σ = 0.05
x₁ = rand(Uniform(-1, 1), n)
x₂ = rand(Normal(1.0, 0.5), n)
X = hcat(ones(n), x₁, x₂, x₁ .* x₂)
ε = rand(Normal(0.0, σ), n)
y = X * β + ε
sse = DiffCache(zeros(n))
β_cache = DiffCache(similar(β), 10)
dat = (y, X, sse, n, β_cache)
@inline function loglik_fnc(θ, data)
    σ, β₀, β₁, β₂, β₃ = θ
    y, X, sse, n, β = data
    _sse = get_tmp(sse, θ)
    _β = get_tmp(β, θ)
    _β[1] = β₀
    _β[2] = β₁
    _β[3] = β₂
    _β[4] = β₃
    ℓℓ = -0.5n * log(2π * σ^2)
    mul!(_sse, X, _β)
    for i in eachindex(y)
        ℓℓ = ℓℓ - 0.5 / σ^2 * (y[i] - _sse[i])^2
    end
    return ℓℓ
end

## Step 2: Define the problem 
θ₀ = ones(5)
prob = LikelihoodProblem(loglik_fnc, θ₀;
    data=dat,
    f_kwargs=(adtype=Optimization.AutoForwardDiff(),),
    prob_kwargs=(
        lb=[0.0, -Inf, -Inf, -Inf, -Inf],
        ub=Inf * ones(5)
    ),
    syms=[:σ, :β₀, :β₁, :β₂, :β₃]
)

## Step 3: Compute the MLE
sol = mle(prob, Optim.LBFGS())
df = n - (length(β) + 1)
resids = y .- X * sol[2:5]
@test sol[2:5] ≈ inv(X' * X) * X' * y # sol[i] = sol.mle[i] 
@test sol[:σ]^2 ≈ 1 / df * sum(resids .^ 2) atol = 1e-4 # symbol indexing

## Step 4: Profile 
lb = [1e-12, -5.0, -5.0, -5.0, -5.0]
ub = [15.0, 15.0, 15.0, 15.0, 15.0]
resolutions = [600, 200, 200, 200, 200] # use many points for σ
param_ranges = construct_profile_ranges(sol, lb, ub, resolutions)
prof1 = profile(prob, sol; param_ranges, parallel=true)
prof2 = profile(prob, sol; param_ranges, parallel=false)

@test prof1.confidence_intervals[1].lower ≈ prof2.confidence_intervals[1].lower rtol = 1e-3
@test prof1.confidence_intervals[2].lower ≈ prof2.confidence_intervals[2].lower rtol = 1e-3
@test prof1.confidence_intervals[3].lower ≈ prof2.confidence_intervals[3].lower rtol = 1e-3
@test prof1.confidence_intervals[4].lower ≈ prof2.confidence_intervals[4].lower rtol = 1e-3
@test prof1.confidence_intervals[1].upper ≈ prof2.confidence_intervals[1].upper rtol = 1e-2
@test prof1.confidence_intervals[2].upper ≈ prof2.confidence_intervals[2].upper rtol = 1e-3
@test prof1.confidence_intervals[3].upper ≈ prof2.confidence_intervals[3].upper rtol = 1e-3
@test prof1.confidence_intervals[4].upper ≈ prof2.confidence_intervals[4].upper rtol = 1e-3
@test prof1.other_mles[1] ≈ prof2.other_mles[1] rtol = 1e-0 atol = 1e-2
@test prof1.other_mles[2] ≈ prof2.other_mles[2] rtol = 1e-1 atol = 1e-2
@test prof1.other_mles[3] ≈ prof2.other_mles[3] rtol = 1e-3 atol = 1e-2
@test prof1.other_mles[4] ≈ prof2.other_mles[4] rtol = 1e-0 atol = 1e-2
@test prof1.parameter_values[1] ≈ prof2.parameter_values[1] rtol = 1e-3
@test prof1.parameter_values[2] ≈ prof2.parameter_values[2] rtol = 1e-3
@test prof1.parameter_values[3] ≈ prof2.parameter_values[3] rtol = 1e-3
@test prof1.parameter_values[4] ≈ prof2.parameter_values[4] rtol = 1e-3
@test issorted(prof1.parameter_values[1])
@test issorted(prof1.parameter_values[2])
@test issorted(prof1.parameter_values[3])
@test issorted(prof1.parameter_values[4])
@test issorted(prof2.parameter_values[1])
@test issorted(prof2.parameter_values[2])
@test issorted(prof2.parameter_values[3])
@test issorted(prof2.parameter_values[4])
@test prof1.profile_values[1] ≈ prof2.profile_values[1] rtol = 1e-1
@test prof1.profile_values[2] ≈ prof2.profile_values[2] rtol = 1e-1
@test prof1.profile_values[3] ≈ prof2.profile_values[3] rtol = 1e-1
@test prof1.profile_values[4] ≈ prof2.profile_values[4] rtol = 1e-1
@test prof1.splines[1].itp.knots ≈ prof2.splines[1].itp.knots
@test prof1.splines[2].itp.knots ≈ prof2.splines[2].itp.knots
@test prof1.splines[3].itp.knots ≈ prof2.splines[3].itp.knots
@test prof1.splines[4].itp.knots ≈ prof2.splines[4].itp.knots

prof = prof1

# Compare the confidence intervals
vcov_mat = sol[:σ]^2 * inv(X' * X)
for i in 1:4
    @test prof.confidence_intervals[i+1][1] ≈ sol.mle[i+1] - 1.96sqrt(vcov_mat[i, i]) atol = 1e-3
    @test prof.confidence_intervals[i+1][2] ≈ sol.mle[i+1] + 1.96sqrt(vcov_mat[i, i]) atol = 1e-3
end
rss = sum(resids .^ 2)
χ²_up = quantile(Chisq(df), 0.975)
χ²_lo = quantile(Chisq(df), 0.025)
σ_CI_exact = sqrt.(rss ./ (χ²_up, χ²_lo))
@test get_confidence_intervals(prof, :σ).lower ≈ σ_CI_exact[1] atol = 1e-3
@test ProfileLikelihood.get_upper(get_confidence_intervals(prof, :σ)) ≈ σ_CI_exact[2] atol = 1e-3

# Can also view a single parameter's results, e.g. 
prof[:β₂] # This is a ProfileLikelihoodSolutionView

# Can also evaluate the profile at a point inside the range. If 
# you want to evaluate outside the confidence interval, you need to 
# use a non-Throw extrap in the profile kwarg (see also Interpolations.jl).
# These are all the same, evaluating the profile for β₂ at β₂=0.50
prof[:β₂](0.50)
prof(0.50, :β₂)
prof(0.50, 4)

## Step 5: Visualise 
using CairoMakie, LaTeXStrings
fig = plot_profiles(prof;
    latex_names=[L"\sigma", L"\beta_0", L"\beta_1", L"\beta_2", L"\beta_3"], # default names would be of the form θᵢ
    show_mles=true,
    shade_ci=true,
    true_vals=[σ, β...],
    fig_kwargs=(fontsize=30, resolution=(2134.0f0, 906.0f0)),
    axis_kwargs=(width=600, height=300))
xlims!(fig.content[1], 0.045, 0.055) # fix the ranges
xlims!(fig.content[2], -1.025, -0.975)
xlims!(fig.content[4], 0.475, 0.525)
SAVE_FIGURE && save("figures/regression_profiles.png", fig)

# You can also plot specific parameters 
plot_profiles(prof, [1, 3]) # plot σ and β₁
plot_profiles(prof, [:σ, :β₁, :β₃]) # can use symbols 
plot_profiles(prof, 1) # can just provide an integer 
plot_profiles(prof, :β₂) # symbols work