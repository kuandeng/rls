include("laser.jl")
include("../grid_cli.jl")
include("../invlan_control_cli.jl")
include("../output_paths.jl")

using Plots

default_grid = (x_st = x_st, x_ed = x_ed, y_st = y_st, y_ed = y_ed, n_x = n_x, n_y = n_y, k = 1)
grid = parse_recursion_grid_args(default_grid)
x_st, x_ed = grid.x_st, grid.x_ed
y_st, y_ed = grid.y_st, grid.y_ed
n_x, n_y = grid.n_x, grid.n_y
k = grid.k
rec = recursion_grid(x_st, x_ed, y_st, y_ed, n_x, n_y, k)

default_control = PsInvLancControl(tol = 1e-4, continue_ = false)
control = parse_ps_inv_lanc_control_args(default_control)
elapsed = @elapsed begin
    σ_min_invlan, stats_invlan = psInvLanc(C, rec; control = control)
end
println("elapsed = ", elapsed, " seconds")
# mv_invlan = stats_invlan.mv

σ_min_invlan_perm = reorder_to_natural(rec, σ_min_invlan)
σ_min_invlan_grid = reorder_to_natural_grid(rec, σ_min_invlan)

output_dir = example_output_dir(@__FILE__)
outpath = joinpath(output_dir, "psInvLanc_stats.jls")
open(outpath, "w") do io
    serialize(io, (
        σ_min_invlan = σ_min_invlan,
        σ_min_invlan_perm = σ_min_invlan_perm,
        σ_min_invlan_grid = σ_min_invlan_grid,
        stats_invlan = stats_invlan,
        elapsed = elapsed,
    ))
end

# x = range(x_st, x_ed; length = n_x)
# y = range(y_st, y_ed; length = n_y)

# plt = contour(
#     x,
#     y,
#     log10.(σ_min_invlan_grid);
#     levels = 20,
#     linewidth = 1.2,
#     fill = true,
#     colorbar_title = "log10(sigma_min)",
#     xlabel = "Re(z)",
#     ylabel = "Im(z)",
#     title = "laser psInvLanc contour"
# )

# png_path = joinpath(output_dir, "psInvLanc_contour.png")
# savefig(plt, png_path)
# println("saved contour to ", png_path)

# time 6198.269651 seconds not continue
