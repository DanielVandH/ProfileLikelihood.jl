######################################################
## IrregularGrid
######################################################
## Test that we are constructing it correctly 
lb = [2.0, 5.0, 1.3]
ub = [5.0, 10.0, 17.3]
grid = [rand(2) for _ in 1:200]
ig = PL.IrregularGrid(lb, ub, grid)
@test PL.get_lower_bounds(ig) == ig.lower_bounds == lb
@test PL.get_upper_bounds(ig) == ig.upper_bounds == ub
@test all(i -> PL.get_lower_bounds(ig, i) == lb[i], eachindex(lb))
@test all(i -> PL.get_upper_bounds(ig, i) == ub[i], eachindex(ub))
@test PL.get_grid(ig) == grid
@test all(i -> PL.get_parameters(grid, i) == ig[i] == grid[i], eachindex(grid))
@test PL.number_type(ig) == Float64
@test PL.number_of_parameter_sets(ig) == 200
@test PL.each_parameter(ig) == 1:200

lb = [2.0, 5.0, 1.3]
ub = [5.0, 10.0, 17.3]
grid = rand(3, 50)
ig = PL.IrregularGrid(lb, ub, grid)
@test PL.get_lower_bounds(ig) == ig.lower_bounds == lb
@test PL.get_upper_bounds(ig) == ig.upper_bounds == ub
@test all(i -> PL.get_lower_bounds(ig, i) == lb[i], eachindex(lb))
@test all(i -> PL.get_upper_bounds(ig, i) == ub[i], eachindex(ub))
@test PL.get_grid(ig) == grid
@test all(i -> PL.get_parameters(grid, i) == ig[i] == grid[:, i], axes(grid, 2))
@test PL.number_type(ig) == Float64
@test PL.number_of_parameter_sets(ig) == 50
@test PL.each_parameter(ig) == 1:50