struct Recursion
    x_st::Float64
    x_ed::Float64
    y_st::Float64
    y_ed::Float64
    n_x::Int
    n_y::Int
    δx::Float64
    δy::Float64
    k::Int
    Z::Vector{ComplexF64}
end

"""
    Recursion(x_st, x_ed, y_st, y_ed, n_x, n_y, k)

Create the grid metadata and allocate storage for the traversal order.
"""
function Recursion(
    x_st::Real, x_ed::Real, y_st::Real, y_ed::Real,
    n_x::Integer, n_y::Integer, k::Integer
)
    nx = Int(n_x)
    ny = Int(n_y)
    kk = Int(k)
    nx >= 2 || throw(ArgumentError("n_x must be >= 2"))
    ny >= 2 || throw(ArgumentError("n_y must be >= 2"))
    kk >= 1 || throw(ArgumentError("k must be >= 1"))

    x1 = Float64(x_st)
    x2 = Float64(x_ed)
    y1 = Float64(y_st)
    y2 = Float64(y_ed)
    δx = (x2 - x1) / (nx - 1)
    δy = (y2 - y1) / (ny - 1)
    Z = Vector{ComplexF64}(undef, nx * ny)
    return Recursion(x1, x2, y1, y2, nx, ny, δx, δy, kk, Z)
end

# Convert integer grid coordinates to the corresponding complex point.
@inline function _z(rec::Recursion, ix::Int, iy::Int)
    x = rec.x_st + (ix - 1) * rec.δx
    y = rec.y_st + (iy - 1) * rec.δy
    return complex(x, y)
end

# Fill one k-point vertical column segment in the recursion traversal.
@inline function _set_col_k!(rec::Recursion, block_id::Int, ix::Int, iy0::Int, down::Bool)
    base = block_id * rec.k
    if down
        @inbounds for l = 1:rec.k
            rec.Z[base + l] = _z(rec, ix, iy0 + l - 1)
        end
    else
        @inbounds for l = 1:rec.k
            rec.Z[base + l] = _z(rec, ix, iy0 + (rec.k - l))
        end
    end
    return nothing
end

"""
    recursion_grid(...)

Default grid order: block zigzag for compatibility with the legacy traversal.
Requires `n_y % k == 0`.
"""
function recursion_grid(
    x_st::Real, x_ed::Real, y_st::Real, y_ed::Real,
    n_x::Integer, n_y::Integer, k::Integer
)
    return recursion_grid_zigzag(x_st, x_ed, y_st, y_ed, n_x, n_y, k)
end

"""
    recursion_grid_linear(...)

Row-wise serpentine traversal:
- odd rows: left -> right
- even rows: right -> left
"""
function recursion_grid_linear(
    x_st::Real, x_ed::Real, y_st::Real, y_ed::Real,
    n_x::Integer, n_y::Integer, k::Integer
)
    rec = Recursion(x_st, x_ed, y_st, y_ed, n_x, n_y, k)
    idx = 1
    @inbounds for iy = 1:rec.n_y
        if isodd(iy)
            for ix = 1:rec.n_x
                rec.Z[idx] = _z(rec, ix, iy)
                idx += 1
            end
        else
            for ix = rec.n_x:-1:1
                rec.Z[idx] = _z(rec, ix, iy)
                idx += 1
            end
        end
    end
    return rec
end

"""
    recursion_grid_spiral(...)

Clockwise spiral traversal starting from the upper-left corner.
"""
function recursion_grid_spiral(
    x_st::Real, x_ed::Real, y_st::Real, y_ed::Real,
    n_x::Integer, n_y::Integer, k::Integer
)
    rec = Recursion(x_st, x_ed, y_st, y_ed, n_x, n_y, k)

    left, right = 1, rec.n_x
    top, bottom = 1, rec.n_y
    idx = 1

    while left <= right && top <= bottom
        @inbounds for ix = left:right
            rec.Z[idx] = _z(rec, ix, top)
            idx += 1
        end
        top += 1
        top > bottom && break

        @inbounds for iy = top:bottom
            rec.Z[idx] = _z(rec, right, iy)
            idx += 1
        end
        right -= 1
        left > right && break

        @inbounds for ix = right:-1:left
            rec.Z[idx] = _z(rec, ix, bottom)
            idx += 1
        end
        bottom -= 1
        top > bottom && break

        @inbounds for iy = bottom:-1:top
            rec.Z[idx] = _z(rec, left, iy)
            idx += 1
        end
        left += 1
    end
    return rec
end

"""
    recursion_grid_zigzag(...)

Block zigzag traversal using groups of `k` rows:
- alternate the x direction for each block row
- alternate the y direction within each small column
"""
function recursion_grid_zigzag(
    x_st::Real, x_ed::Real, y_st::Real, y_ed::Real,
    n_x::Integer, n_y::Integer, k::Integer
)
    rec = Recursion(x_st, x_ed, y_st, y_ed, n_x, n_y, k)
    rec.n_y % rec.k == 0 || throw(ArgumentError("n_y must be a multiple of k"))

    n_block_rows = rec.n_y ÷ rec.k
    block_id = 0
    for by = 1:n_block_rows
        x_range = isodd(by) ? (1:rec.n_x) : (rec.n_x:-1:1)
        iy0 = (by - 1) * rec.k + 1
        for (j, ix) in enumerate(x_range)
            down = isodd(j)
            _set_col_k!(rec, block_id, ix, iy0, down)
            block_id += 1
        end
    end
    return rec
end

# Map a grid point back to its row-major natural-order index.
@inline function _natural_linear_index(rec::Recursion, z::ComplexF64)
    rx = (real(z) - rec.x_st) / rec.δx
    ry = (imag(z) - rec.y_st) / rec.δy
    ix = round(Int, rx) + 1
    iy = round(Int, ry) + 1
    (1 <= ix <= rec.n_x && 1 <= iy <= rec.n_y) || throw(
        ArgumentError("point $(z) is outside the recursion grid bounds")
    )
    return (iy - 1) * rec.n_x + ix
end

"""
    recursion_natural_permutation(rec)

Return a permutation `perm` such that `rec.Z[perm]` is in natural order:
outer loop `iy=1:n_y`, inner loop `ix=1:n_x`, left to right by row.
"""
function recursion_natural_permutation(rec::Recursion)
    n = rec.n_x * rec.n_y
    perm = zeros(Int, n)
    @inbounds for i = 1:n
        j = _natural_linear_index(rec, rec.Z[i])
        perm[j] == 0 || throw(ArgumentError("duplicate grid point detected at rec.Z[$i]"))
        perm[j] = i
    end
    @inbounds for j = 1:n
        perm[j] > 0 || throw(ArgumentError("missing grid point for natural index $j"))
    end
    return perm
end

"""
    reorder_to_natural(rec, v)

Reorder a vector `v` stored in the current `rec.Z` order into natural order.
This is consistent with `recursion_natural_permutation`.
"""
function reorder_to_natural(rec::Recursion, v::AbstractVector)
    n = rec.n_x * rec.n_y
    length(v) == n || throw(DimensionMismatch("v length $(length(v)) does not match grid size $n"))
    perm = recursion_natural_permutation(rec)
    return v[perm]
end

"""
    reorder_to_natural_grid(rec, v)

Reorder a vector `v` stored in the current `rec.Z` order into an `n_y x n_x` matrix.
`A[iy, ix]` corresponds to grid point `(x[ix], y[iy])` and can be used directly for contours.
"""
function reorder_to_natural_grid(rec::Recursion, v::AbstractVector)
    v_nat = reorder_to_natural(rec, v)
    return permutedims(reshape(v_nat, rec.n_x, rec.n_y), (2, 1))
end
