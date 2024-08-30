using ProfileLikelihood
using Test
using SafeTestsets

@testset "ProfileLikelihood" begin
    @safetestset "Utilities" begin
        include("utils.jl")
    end
    @safetestset "Problem updates" begin
        include("problem_updates.jl")
    end
    @safetestset "LikelihoodProblem" begin
        include("likelihood_problem.jl")
    end
    @safetestset "Custom Hessian" begin
        include("custom_hessian.jl")
    end
    @safetestset "MLE" begin
        include("mle.jl")
    end
    @safetestset "RegularGrid" begin
        include("regular_grid.jl")
    end
    @safetestset "IrregularGrid" begin
        include("irregular_grid.jl")
    end
    @safetestset "GridSearch" begin
        include("grid_search.jl")
    end
    @safetestset "ConfidenceInterval" begin
        include("confidence_interval.jl")
    end
    @safetestset "ProfileLikelihood" begin
        include("profile_likelihood.jl")
    end
    @safetestset "Prediction Intervals" begin
        include("prediction_intervals.jl")
    end
    @safetestset "Bivariate ProfileLikelihood" begin
        include("bivariate_profile.jl")
    end
    @safetestset "Example I: Regression" begin
        include("regression_example.jl")
    end
    @safetestset "Example II: Logistic ODE" begin
        include("logistic.jl")
    end
    @safetestset "Example III: Linear Exponential" begin
        include("linear_exponential_example.jl")
    end
    @safetestset "Evample IV: Lotka-Volterra ODE" begin
        include("lotka_volterra_example.jl")
    end
    if !(get(ENV, "CI", "false") == "true")
        @safetestset "Example V: Fisher-Stefan PDE" begin
            include("pde_example.jl")
        end
    end
end