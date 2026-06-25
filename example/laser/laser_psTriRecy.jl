
using Random

function parse_seed_arg()
    for arg in ARGS
        occursin("=", arg) || continue
        key_src, value_src = split(arg, "="; limit = 2)
        if strip(key_src) == "seed"
            seed = parse(Int, strip(value_src))
            Random.seed!(seed)
            println("[laser_tri] seed = ", seed)
            flush(stdout)
            return seed
        end
    end
    return nothing
end

parse_seed_arg()

include("laser.jl")
include("../grid_cli.jl")
include("../tri_control_cli.jl")
include("../output_paths.jl")

using Plots

default_grid = (x_st = -1.1, x_ed = 1.2, y_st = -1.1, y_ed = 1.1, n_x = 200, n_y = 200, k = 20)
grid = parse_recursion_grid_args(default_grid)
x_st, x_ed = grid.x_st, grid.x_ed
y_st, y_ed = grid.y_st, grid.y_ed
n_x, n_y = grid.n_x, grid.n_y
k = grid.k
rec = recursion_grid(x_st, x_ed, y_st, y_ed, n_x, n_y, k)

default_control = PsTriRecyControl(normC = 1.0, tol = 1e-4, recy_tau = 1.6, lobpcg_tol_scale = 0.1, p=60, implict_recycle = true)
control = parse_ps_tri_recy_control_args(default_control)
elapsed = @elapsed begin
    σ_min, stats = psTriRecy(C, rec; control = control)
end
println("elapsed = ", elapsed, " seconds")

σ_min_perm = reorder_to_natural(rec, σ_min)
σ_min_grid = reorder_to_natural_grid(rec, σ_min)

output_dir = example_output_dir(@__FILE__)
outpath = joinpath(output_dir, "psTriRecy_stats.jls")
open(outpath, "w") do io
    serialize(io, (
        σ_min = σ_min,
        σ_min_perm = σ_min_perm,
        σ_min_grid = σ_min_grid,
        stats = stats,
        elapsed = elapsed,
    ))
end

x = range(x_st, x_ed; length = n_x)
y = range(y_st, y_ed; length = n_y)

plt = contour(
    x,
    y,
    log10.(σ_min_grid);
    levels = 20,
    linewidth = 1.2,
    fill = true,
    colorbar_title = "log10(sigma_min)",
    xlabel = "Re(z)",
    ylabel = "Im(z)",
    title = "laser psTriRecy contour"
)

png_path = joinpath(output_dir, "psTriRecy_contour.png")
savefig(plt, png_path)
println("saved contour to ", png_path)

