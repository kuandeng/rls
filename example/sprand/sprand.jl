include("../../src/RLS.jl")
using KrylovKit

function to_int32_index_sparse(C::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti<:Integer}
    m, n = size(C)
    m <= typemax(Int32) || throw(ArgumentError("row size $m exceeds Int32 range"))
    n <= typemax(Int32) || throw(ArgumentError("col size $n exceeds Int32 range"))
    return SparseMatrixCSC{Tv, Int32}(C)
end

N = 4000  # Set problem size here.

Random.seed!(20260318)

d0 = 3 .* exp.(-(0:N-1) ./ 10)         # Main diagonal, length N.
d1 = N > 1 ? fill(0.5, N-1) : Float64[] # First superdiagonal, length N-1.

C = spdiagm(0 => d0, 1 => d1) + 0.1 * sprandn(N, N, 3 / N)

C = ComplexF64.(C)

# σ, _, _, _ = svdsolve(C, 1, :LR)
C = to_int32_index_sparse(C)
# normC = σ[1]
# println("normC (KrylovKit.svdsolve) = ", normC)

normC = 3.2

x_st, x_ed = 0.450000, 0.550000
y_st, y_ed = 0.000000, 0.100000
n_x, n_y = 200, 200
k = 20
# rec = recursion_grid(x_st, x_ed, y_st, y_ed, n_x, n_y, k)
