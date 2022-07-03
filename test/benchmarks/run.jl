using Pkg
Pkg.activate("C:/Users/licer/.julia/dev/ProfileLikelihood")
using ProfileLikelihood
Pkg.activate("C:/Users/licer/.julia/dev/ProfileLikelihood/test")
include("C:/Users/licer/.julia/dev/ProfileLikelihood/test/template_functions.jl")
const CHECK_TYPE = false

using BenchmarkTools
using Optimization
using OptimizationNLopt
using OptimizationOptimJL
using BenchmarkTools: prettytime, prettymemory

const ALG = LBFGS()
const ALG2 = NLopt.LN_NELDERMEAD()

function run_benchmark(problem)
    prob, _ = problem()
    
    ## Objective function 
    θ = prob.θ₀
    p = ProfileLikelihood.data(prob)
    objective_bm = @benchmark $prob.prob.f($θ, $p)
    alg = problem == MultipleLinearRegression ? ALG : ALG2
    
    ## MLE 
    sol = mle(prob, alg)
    @inferred mle(prob, alg)
    CHECK_TYPE && @code_warntype mle(prob, alg)
    mle_bm = @benchmark mle($prob, $alg)

    sol = solve(prob.prob, alg)
    @inferred solve(prob.prob, alg)
    CHECK_TYPE && @code_warntype solve(prob.prob, alg)
    optsol_bm = @benchmark solve($prob.prob, $alg)

    likesol = LikelihoodSolution(sol, prob; alg=alg)
    @inferred LikelihoodSolution(sol, prob; alg=alg)
    CHECK_TYPE && @code_warntype LikelihoodSolution(sol, prob; alg=alg)
    likesol_bm = @benchmark LikelihoodSolution($sol, $prob; alg=$alg)

    mlesol = mle(prob, alg)

    ## Profile 
    sol = mlesol
    min_steps = 15
    max_steps = 100
    Δθ = abs.(mle(sol) / 100)
    alg = alg
    conf_level = 0.95
    spline = true
    threshold = -0.5quantile(Chisq(1), conf_level)

    prof = profile(prob, sol; min_steps, max_steps, Δθ, alg, conf_level, spline, threshold)
    @inferred profile(prob, sol; min_steps, max_steps, Δθ, alg, conf_level, spline, threshold)
    CHECK_TYPE && @code_warntype profile(prob, sol; min_steps, max_steps, Δθ, alg, conf_level, spline, threshold)
    prof_bm = @benchmark profile($prob, $sol; min_steps=$min_steps,
        max_steps=$max_steps,
        Δθ=$Δθ,
        alg=$alg,
        conf_level=$conf_level,
        spline=$spline,
        threshold=$threshold)

    ## Single profile

    N = ProfileLikelihood.num_params(prob)
    if Δθ isa Number
        Δθ = repeat([Δθ], N)
    end
    θ = Dict{Int64,Float64}([])
    prof = Dict{Int64,Vector{Float64}}([])
    splines = Vector{Spline1D}([])
    sizehint!(θ, N)
    sizehint!(prof, N)
    sizehint!(splines, N)

    n = 2
    profile_vals, param_vals = profile(prob, sol, n; min_steps, max_steps, threshold, Δθ=Δθ[n], alg)
    @inferred profile(prob, sol, n; min_steps, max_steps, threshold, Δθ=Δθ[n], alg)
    CHECK_TYPE && @code_warntype profile(prob, sol, n; min_steps, max_steps, threshold, Δθ=Δθ[n], alg)
    profsingle_bm = @benchmark profile($prob, $sol, $n;
        min_steps=$min_steps,
        max_steps=$max_steps,
        threshold=$threshold,
        Δθ=$Δθ[$n],
        alg=$alg)

    profile_vals, param_vals = profile(prob.prob, sol.θ, sol.maximum, n, min_steps, max_steps, threshold, Δθ[n], alg)
    @inferred profile(prob.prob, sol.θ, sol.maximum, n, min_steps, max_steps, threshold, Δθ[n], alg)
    CHECK_TYPE && @code_warntype profile(prob.prob, sol.θ, sol.maximum, n, min_steps, max_steps, threshold, Δθ[n], alg)
    profsingleopt_bm = @benchmark profile($prob.prob, $sol.θ, $sol.maximum, $n, $min_steps, $max_steps, $threshold, $Δθ[$n], $alg)

    ## Step preparation and problem updates

    θₘₗₑ = sol.θ
    ℓₘₐₓ = sol.maximum
    i = n
    Δ = Δθ[i]

    Prob, param_vals, profile_vals, cache, θ₀ = ProfileLikelihood.prepare_profile(prob.prob, θₘₗₑ, max_steps, i)
    @inferred ProfileLikelihood.prepare_profile(prob.prob, θₘₗₑ, max_steps, i)
    CHECK_TYPE && @code_warntype ProfileLikelihood.prepare_profile(prob.prob, θₘₗₑ, max_steps, i)
    prepareprofile_bm = @benchmark ProfileLikelihood.prepare_profile($prob.prob, $θₘₗₑ, $max_steps, $i)

    Prob = ProfileLikelihood.update_prob(prob.prob, i)
    @inferred ProfileLikelihood.update_prob(prob.prob, i)
    CHECK_TYPE && @code_warntype ProfileLikelihood.update_prob(prob.prob, i)
    removeboundupdate_bm = @benchmark ProfileLikelihood.update_prob($prob.prob, $i)

    Prob = ProfileLikelihood.update_prob(prob.prob, θ₀)
    @inferred ProfileLikelihood.update_prob(prob.prob, θ₀)
    CHECK_TYPE && @code_warntype ProfileLikelihood.update_prob(prob.prob, θ₀)
    setinitial_bm = @benchmark ProfileLikelihood.update_prob($prob.prob, $θ₀)

    Prob = ProfileLikelihood.update_prob(Prob, i, θₘₗₑ[i], cache)
    @inferred ProfileLikelihood.update_prob(Prob, i, θₘₗₑ[i], cache)
    CHECK_TYPE && @code_warntype ProfileLikelihood.update_prob(Prob, i, θₘₗₑ[i], cache)
    setnewobj_bm = @benchmark ProfileLikelihood.update_prob($Prob, $i, $θₘₗₑ[$i], $cache)

    ff = ProfileLikelihood.construct_new_f(Prob, i, θₘₗₑ[i], cache)
    @inferred ProfileLikelihood.construct_new_f(Prob, i, θₘₗₑ[i], cache)
    CHECK_TYPE && @code_warntype ProfileLikelihood.construct_new_f(Prob, i, θₘₗₑ[i], cache)
    constructnewobj_bm = @benchmark ProfileLikelihood.construct_new_f($Prob, $i, $θₘₗₑ[$i], $cache)

    Prob = ProfileLikelihood.update_prob(Prob, i, θₘₗₑ[i], cache, θ₀)
    @inferred ProfileLikelihood.update_prob(Prob, i, θₘₗₑ[i], cache, θ₀)
    CHECK_TYPE && @code_warntype ProfileLikelihood.update_prob(Prob, i, θₘₗₑ[i], cache, θ₀)
    constructnewobjandinit_bm = @benchmark ProfileLikelihood.update_prob($Prob, $i, $θₘₗₑ[$i], $cache, $θ₀)

    ## Stepping 

    Prob, param_vals, profile_vals, cache, θ₀ = ProfileLikelihood.prepare_profile(prob.prob, θₘₗₑ, max_steps, i)
    ProfileLikelihood.profile!(Prob, ℓₘₐₓ, i, θₘₗₑ[i], param_vals, profile_vals, θ₀, cache; alg)
    CHECK_TYPE && @code_warntype ProfileLikelihood.profile!(Prob, ℓₘₐₓ, i, θₘₗₑ[i], param_vals, profile_vals, θ₀, cache; alg)
    singlestepatmle_bm = @benchmark ProfileLikelihood.profile!($Prob, $ℓₘₐₓ, $i, $θₘₗₑ[$i], $param_vals, $profile_vals, $θ₀, $cache; alg=$alg)

    ProfileLikelihood.step_profile!(Prob, ℓₘₐₓ, i, param_vals, profile_vals, θ₀, cache; alg, Δθ=Δθ[i])
    CHECK_TYPE && @code_warntype ProfileLikelihood.step_profile!(Prob, ℓₘₐₓ, i, param_vals, profile_vals, θ₀, cache; alg, Δθ=Δθ[i])
    singlestepprofile_bm = @benchmark ProfileLikelihood.step_profile!($Prob, $ℓₘₐₓ, $i, $param_vals, $profile_vals, $θ₀, $cache; alg=$alg, Δθ=$Δθ[$i])

    ProfileLikelihood.find_endpoint!(Prob, profile_vals, threshold, min_steps, max_steps, ℓₘₐₓ, i, param_vals, θ₀, cache, alg, -Δθ[i])
    CHECK_TYPE && @code_warntype ProfileLikelihood.find_endpoint!(Prob, profile_vals, threshold, min_steps, max_steps, ℓₘₐₓ, i, param_vals, θ₀, cache, alg, -Δθ[i])
    Prob, param_vals, profile_vals, cache, θ₀ = ProfileLikelihood.prepare_profile(prob.prob, θₘₗₑ, max_steps, i)
    ProfileLikelihood.profile!(Prob, ℓₘₐₓ, i, θₘₗₑ[i], param_vals, profile_vals, θ₀, cache; alg)
    findendpoint_bm = @benchmark ProfileLikelihood.find_endpoint!($Prob, deepcopy($profile_vals), $threshold, $min_steps, $max_steps, $ℓₘₐₓ, $i, deepcopy($param_vals), $θ₀, $cache, $alg, -$Δθ[$i])

    ## Confidence intervals 
    prof = profile(prob, sol; min_steps, max_steps, Δθ, alg, conf_level, spline, threshold)

    N = ProfileLikelihood.num_params(prob)
    if Δθ isa Number
        Δθ = repeat([Δθ], N)
    end
    θ = Dict{Int64,Vector{Float64}}([])
    prof = Dict{Int64,Vector{Float64}}([])
    splines = Vector{Spline1D}([])
    sizehint!(θ, N)
    sizehint!(prof, N)
    sizehint!(splines, N)
    for n in 1:N
        profile_vals, param_vals = profile(prob, sol, n; min_steps, max_steps, threshold, Δθ=Δθ[n], alg)
        θ[n] = param_vals
        prof[n] = profile_vals
        push!(splines, Spline1D(param_vals, profile_vals))
    end
    splines = Dict(1:N .=> splines) # We define the Dict here rather than above to make sure we get the types right
    conf_ints = confidence_intervals(θ, prof; conf_level, spline)
    @inferred confidence_intervals(θ, prof; conf_level, spline)
    CHECK_TYPE && @code_warntype confidence_intervals(θ, prof; conf_level, spline)
    confints_bm = @benchmark confidence_intervals($θ, $prof; conf_level=$conf_level, spline=$spline)

    confintssingle = confidence_intervals(θ, prof, i; conf_level, spline)
    @inferred confidence_intervals(θ, prof, i; conf_level, spline)
    CHECK_TYPE && @code_warntype confidence_intervals(θ, prof, i; conf_level, spline)
    confintssingle_bm = @benchmark confidence_intervals($θ, $prof, $i; conf_level=$conf_level, spline=$spline)

    ## Solution 
    profile_sol = ProfileLikelihoodSolution(θ, prof, prob, sol, splines, conf_ints)
    @inferred ProfileLikelihoodSolution(θ, prof, prob, sol, splines, conf_ints)
    CHECK_TYPE && @code_warntype ProfileLikelihoodSolution(θ, prof, prob, sol, splines, conf_ints)
    solconstruct_bm = @benchmark ProfileLikelihoodSolution($θ, $prof, $prob, $sol, $splines, $conf_ints)

    ## Return 
    objfnc = [objective_bm]
    mlefnc = [mle_bm, optsol_bm, likesol_bm]
    completeproffnc = [prof_bm]
    singleproffnc = [profsingle_bm, profsingleopt_bm]
    problemupdatefnc = [prepareprofile_bm, removeboundupdate_bm, setinitial_bm, setnewobj_bm, constructnewobj_bm, constructnewobjandinit_bm]
    steppingfnc = [singlestepatmle_bm, singlestepprofile_bm, findendpoint_bm]
    confintfnc = [confints_bm, confintssingle_bm]
    solconstructfnc = [solconstruct_bm]

    return objfnc, mlefnc, completeproffnc, singleproffnc, problemupdatefnc, steppingfnc, confintfnc, solconstructfnc
end

objfnc_names = ["Objective (Optimization)"]
mlefnc_names = ["`mle`", "`Optimization.solve`", "`LikelihoodSolution`"]
completeproffnc_names = ["`profile` (all variables)"]
singleproffnc_names = ["`profile` (one variable)", "`profile` (one variable, lowered)"]
problemupdatefnc_names = ["`prepare_profile`", "`update_prob` (remove bounds)", "`update_prob` (set initial guess)", "`update_prob` (set new objective)", "`construct_new_f`", "`update_prob` (set initial guess and new objective)"]
steppingfnc_names = ["`profile!`", "`step_profile!`", "`find_endpoint!`"]
confintfnc_names = ["`confidence_intervals` (all)", "`confidence_intervals` (single)"]
solconstructfnc_names = ["`ProfileLikelihoodSolution`"]


objfnc_mlr, mlefnc_mlr, completeproffnc_mlr, singleproffnc_mlr,
problemupdatefnc_mlr, steppingfnc_mlr, confintfnc_mlr, solconstructfnc_mlr = run_benchmark(MultipleLinearRegression)
objfnc_linear, mlefnc_linear, completeproffnc_linear, singleproffnc_linear,
problemupdatefnc_linear, steppingfnc_linear, confintfnc_linear, solconstructfnc_linear = run_benchmark(LinearExponentialODE)
objfnc_logistic, mlefnc_logistic, completeproffnc_logistic, singleproffnc_logistic,
problemupdatefnc_logistic, steppingfnc_logistic, confintfnc_logistic, solconstructfnc_logistic = run_benchmark(LogisticODE)

all_res = [(objfnc_mlr, objfnc_linear, objfnc_logistic),
    (mlefnc_mlr, mlefnc_linear, mlefnc_logistic),
    (completeproffnc_mlr, completeproffnc_linear, completeproffnc_logistic),
    (singleproffnc_mlr, singleproffnc_linear, singleproffnc_logistic),
    (problemupdatefnc_mlr, problemupdatefnc_linear, problemupdatefnc_logistic),
    (steppingfnc_mlr, steppingfnc_linear, steppingfnc_logistic),
    (confintfnc_mlr, confintfnc_linear, confintfnc_logistic),
    (solconstructfnc_mlr, solconstructfnc_linear, solconstructfnc_logistic)
]
all_names = [objfnc_names, mlefnc_names, completeproffnc_names, singleproffnc_names,
    problemupdatefnc_names, steppingfnc_names, confintfnc_names, solconstructfnc_names]

open("C:/Users/licer/.julia/dev/ProfileLikelihood/test/benchmarks/README.md", "w") do fout
    write(fout, "# Optimiser benchmarks\n")
    write(fout, "## Basics\n")
    write(
        fout,
        "The table below presents a series of benchmarks for functions that go into 
optimising and profiling a likelihood. We consider three different problems:\n"
    )
    write(fout, "   - Regression: This is a multiple linear regression problem. There are five parameters and    \$n = 300 \$ data points.\n")
    write(fout, "   - Linear exponential ODE: This is related to the problem     \$\\mathrm dy/\\mathrm dt = \\lambda y \$ with initial condition   \$ y(0) = y_0   \$. There are three parameters and \$ n = 200\$ data points.\n")
    write(fout, "   - Logistic ODE: This is related to the problem     \$\\mathrm du/\\mathrm dt = \\lambda u (1 - u/K)\$ with initial condition   \$u(0) = u_0  \$.    There are four parameters and   \$ n = 100\$ data points.\n\n")
    write(fout, "| Function | Problem | Median Time | Mean Time | Memory | Allocations |\n")
    write(fout, "|---|---|--:|--:|--:|--:|\n")
    for (j, (function_names, results)) in enumerate(zip(all_names, all_res))
        if j > 1
            write(fout, "| | | | | | |\n")
        end
        for i in eachindex(function_names)
            fnc = function_names[i]
            res = results[1][i], results[2][i], results[3][i]
            med₁, med₂, med₃ = @. prettytime(time(median(res)))
            mean₁, mean₂, mean₃ = @. prettytime(time(mean(res)))
            mem₁, mem₂, mem₃ = @. string(prettymemory(memory(minimum(res))))
            alloc₁, alloc₂, alloc₃ = @. string(allocs(minimum(res)))
            write(fout, "| $fnc | Regression<br>Linear exponential ODE<br>Logistic ODE | $med₁<br>$med₂<br>$med₃ | $mean₁<br>$mean₂<br>$mean₃ | $mem₁<br>$mem₂<br>$mem₃ | $alloc₁<br>$alloc₂<br>$alloc₃ |\n")
        end
    end
end



