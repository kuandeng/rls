function _parse_grid_value(value_src::AbstractString)
    return Core.eval(@__MODULE__, Meta.parse(strip(value_src)))
end

function parse_recursion_grid_args(default_grid::NamedTuple)
    haskey(default_grid, :x_st) || throw(ArgumentError("default_grid must include x_st"))
    haskey(default_grid, :x_ed) || throw(ArgumentError("default_grid must include x_ed"))
    haskey(default_grid, :y_st) || throw(ArgumentError("default_grid must include y_st"))
    haskey(default_grid, :y_ed) || throw(ArgumentError("default_grid must include y_ed"))
    haskey(default_grid, :n_x) || throw(ArgumentError("default_grid must include n_x"))
    haskey(default_grid, :n_y) || throw(ArgumentError("default_grid must include n_y"))

    grid = Dict{Symbol, Any}(
        :x_st => Float64(default_grid.x_st),
        :x_ed => Float64(default_grid.x_ed),
        :y_st => Float64(default_grid.y_st),
        :y_ed => Float64(default_grid.y_ed),
        :n_x => Int(default_grid.n_x),
        :n_y => Int(default_grid.n_y),
    )

    if haskey(default_grid, :k)
        grid[:k] = Int(default_grid.k)
    end

    for arg in ARGS
        occursin("=", arg) || continue
        parts = split(arg, "="; limit = 2)
        length(parts) == 2 || continue
        key = Symbol(strip(parts[1]))
        value_src = strip(parts[2])

        if key in (:x_st, :x_ed, :y_st, :y_ed)
            grid[key] = Float64(_parse_grid_value(value_src))
        elseif key in (:n_x, :n_y, :k)
            grid[key] = Int(_parse_grid_value(value_src))
        end
    end

    return haskey(grid, :k) ? (
        x_st = grid[:x_st],
        x_ed = grid[:x_ed],
        y_st = grid[:y_st],
        y_ed = grid[:y_ed],
        n_x = grid[:n_x],
        n_y = grid[:n_y],
        k = grid[:k],
    ) : (
        x_st = grid[:x_st],
        x_ed = grid[:x_ed],
        y_st = grid[:y_st],
        y_ed = grid[:y_ed],
        n_x = grid[:n_x],
        n_y = grid[:n_y],
    )
end
