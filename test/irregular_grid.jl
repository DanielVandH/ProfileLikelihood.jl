using ..ProfileLikelihood

@testset "Test that we are constructing the IrregularGrid correctly" begin
    lb = [2.0, 5.0, 1.3]
    ub = [5.0, 10.0, 17.3]
    grid = [rand(2) for _ in 1:200]
    ig = ProfileLikelihood.IrregularGrid(lb, ub, grid)
    @test ProfileLikelihood.get_lower_bounds(ig) == ig.lower_bounds == lb
    @test ProfileLikelihood.get_upper_bounds(ig) == ig.upper_bounds == ub
    @test all(i -> ProfileLikelihood.get_lower_bounds(ig, i) == lb[i], eachindex(lb))
    @test all(i -> ProfileLikelihood.get_upper_bounds(ig, i) == ub[i], eachindex(ub))
    @test ProfileLikelihood.get_grid(ig) == grid
    @test all(i -> ProfileLikelihood.get_parameters(grid, i) == ig[i] == grid[i], eachindex(grid))
    @test ProfileLikelihood.number_type(ig) == Float64
    @test ProfileLikelihood.number_of_parameter_sets(ig) == 200
    @test ProfileLikelihood.each_parameter(ig) == 1:200

    lb = [2.0, 5.0, 1.3]
    ub = [5.0, 10.0, 17.3]
    grid = rand(3, 50)
    ig = ProfileLikelihood.IrregularGrid(lb, ub, grid)
    @test ProfileLikelihood.get_lower_bounds(ig) == ig.lower_bounds == lb
    @test ProfileLikelihood.get_upper_bounds(ig) == ig.upper_bounds == ub
    @test all(i -> ProfileLikelihood.get_lower_bounds(ig, i) == lb[i], eachindex(lb))
    @test all(i -> ProfileLikelihood.get_upper_bounds(ig, i) == ub[i], eachindex(ub))
    @test ProfileLikelihood.get_grid(ig) == grid
    @test all(i -> ProfileLikelihood.get_parameters(grid, i) == ig[i] == grid[:, i], axes(grid, 2))
    @test ProfileLikelihood.number_type(ig) == Float64
    @test ProfileLikelihood.number_of_parameter_sets(ig) == 50
    @test ProfileLikelihood.each_parameter(ig) == 1:50
end