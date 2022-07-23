using Pkg
Pkg.activate("C:/Users/licer/.julia/dev/ProfileLikelihood")
using ProfileLikelihood
Pkg.activate("C:/Users/licer/.julia/dev/ProfileLikelihood/test")
include("C:/Users/licer/.julia/dev/ProfileLikelihood/test/template_functions.jl")
const CHECK_TYPE = true

using BenchmarkTools
using Optimization
using OptimizationNLopt
using OptimizationOptimJL
using BenchmarkTools: prettytime, prettymemory

const ALG = LBFGS()
const ALG2 = NLopt.LN_NELDERMEAD()

function run_benchmark(problem, alg)
    prob, _ = problem()

    ## Objective function 
    θ = prob.θ₀
    p = ProfileLikelihood.data(prob)
    objective_bm = @benchmark $prob.prob.f($θ, $p)

    ## MLE 
    mlesol = mle(prob, alg)
    mle_bm = @benchmark mle($prob, $alg)

    ## Profile 
    if problem == MultipleLinearRegression
        resolution = 1000
        param_bounds = [
            (0.001, 0.1),
            (-1.2, -0.8),
            (0.8, 1.2),
            (0.3, 0.7),
            (2.5, 3.5)
        ]
        param_ranges = ProfileLikelihood.construct_profile_ranges(prob, mlesol, resolution; param_bounds)
        prof_bm = @benchmark profile($prob, $mlesol; alg=$alg, param_ranges=$param_ranges)
    else
        resolution = 1000
        param_ranges = ProfileLikelihood.construct_profile_ranges(prob, mlesol, resolution)
        prof_bm = @benchmark profile($prob, $mlesol; alg=$alg, param_ranges=$param_ranges)
    end
    ## Return 
    #objfnc = [objective_bm]
    #mlefnc = [mle_bm, optsol_bm, likesol_bm]
    #completeproffnc = [prof_bm]
    #singleproffnc = [profsingle_bm, profsingleopt_bm]
    #problemupdatefnc = [prepareprofile_bm, removeboundupdate_bm, setinitial_bm, setnewobj_bm, constructnewobj_bm, constructnewobjandinit_bm]
    #steppingfnc = [singlestepatmle_bm, singlestepprofile_bm, findendpoint_bm]
    #confintfnc = [confints_bm, confintssingle_bm]
    #solconstructfnc = [solconstruct_bm]
    #return objfnc, mlefnc, completeproffnc, singleproffnc, problemupdatefnc, steppingfnc, confintfnc, solconstructfnc
    return objective_bm, mle_bm, prof_bm
end

#objfnc_names = ["Objective (Optimization)"]
#mlefnc_names = ["`mle`", "`Optimization.solve`", "`LikelihoodSolution`"]
#completeproffnc_names = ["`profile` (all variables)"]
#singleproffnc_names = ["`profile` (one variable)", "`profile` (one variable, lowered)"]
#problemupdatefnc_names = ["`prepare_profile`", "`update_prob` (remove bounds)", "`update_prob` (set initial guess)", "`update_prob` (set new objective)", "`construct_new_f`", "`update_prob` (set initial guess and new objective)"]
#steppingfnc_names = ["`profile!`", "`step_profile!`", "`find_endpoint!`"]
#confintfnc_names = ["`confidence_intervals` (all)", "`confidence_intervals` (single)"]
#solconstructfnc_names = ["`ProfileLikelihoodSolution`"]
objfnc_name = ["Objective"]
mlefnc_name = ["`mle`"]
profile_name = ["`profile`"]

#=
objfnc_mlr, mlefnc_mlr, completeproffnc_mlr, singleproffnc_mlr,
problemupdatefnc_mlr, steppingfnc_mlr, confintfnc_mlr, solconstructfnc_mlr = run_benchmark(MultipleLinearRegression)
objfnc_linear, mlefnc_linear, completeproffnc_linear, singleproffnc_linear,
problemupdatefnc_linear, steppingfnc_linear, confintfnc_linear, solconstructfnc_linear = run_benchmark(LinearExponentialODE)
objfnc_logistic, mlefnc_logistic, completeproffnc_logistic, singleproffnc_logistic,
problemupdatefnc_logistic, steppingfnc_logistic, confintfnc_logistic, solconstructfnc_logistic = run_benchmark(LogisticODE)
=#
objfnc_mlr, mlefnc_mlr, completeproffnc_mlr = run_benchmark(MultipleLinearRegression, ALG)
objfnc_linear, mlefnc_linear, completeproffnc_linear = run_benchmark(LinearExponentialODE, ALG2)
objfnc_logistic, mlefnc_logistic, completeproffnc_logistic = run_benchmark(LogisticODE, ALG2)
objfnc_linearad, mlefnc_linearad, completeproffnc_linearad = run_benchmark(LinearExponentialODEAutoDiff, ALG)
#=
all_res = [(objfnc_mlr, objfnc_linear, objfnc_logistic),
    (mlefnc_mlr, mlefnc_linear, mlefnc_logistic),
    (completeproffnc_mlr, completeproffnc_linear, completeproffnc_logistic),
    (singleproffnc_mlr, singleproffnc_linear, singleproffnc_logistic),
    (problemupdatefnc_mlr, problemupdatefnc_linear, problemupdatefnc_logistic),
    (steppingfnc_mlr, steppingfnc_linear, steppingfnc_logistic),
    (confintfnc_mlr, confintfnc_linear, confintfnc_logistic),
    (solconstructfnc_mlr, solconstructfnc_linear, solconstructfnc_logistic)
]
=#
all_res = [(objfnc_mlr, objfnc_linear, objfnc_logistic, objfnc_linearad),
    (mlefnc_mlr, mlefnc_linear, mlefnc_logistic, mlefnc_linearad),
    (completeproffnc_mlr, completeproffnc_linear, completeproffnc_logistic, completeproffnc_linearad),
]
#=
all_names = [objfnc_names, mlefnc_names, completeproffnc_names, singleproffnc_names,
    problemupdatefnc_names, steppingfnc_names, confintfnc_names, solconstructfnc_names]
=#
all_names = [objfnc_name, mlefnc_name, profile_name]

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
            res = results[1][i], results[2][i], results[3][i], results[4][i]
            med₁, med₂, med₃, med₄ = @. prettytime(time(median(res)))
            mean₁, mean₂, mean₃, mean₄ = @. prettytime(time(mean(res)))
            mem₁, mem₂, mem₃, mem₄ = @. string(prettymemory(memory(minimum(res))))
            alloc₁, alloc₂, alloc₃, alloc₄ = @. string(allocs(minimum(res)))
            write(fout, "| $fnc | Regression<br>Linear exponential ODE<br>Logistic ODE<br>Linear exponential ODE (AutoDiff) | $med₁<br>$med₂<br>$med₃<br>$med₄ | $mean₁<br>$mean₂<br>$mean₃<br>$mean₄ | $mem₁<br>$mem₂<br>$mem₃<br>$mem₄ | $alloc₁<br>$alloc₂<br>$alloc₃<br>$alloc₄ |\n")
        end
    end
end



