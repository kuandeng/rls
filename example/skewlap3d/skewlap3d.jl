include("../../src/RLS.jl")
using KrylovKit

function to_int32_index_sparse(C::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti<:Integer}
    m, n = size(C)
    m <= typemax(Int32) || throw(ArgumentError("row size $m exceeds Int32 range"))
    n <= typemax(Int32) || throw(ArgumentError("col size $n exceeds Int32 range"))
    return SparseMatrixCSC{Tv, Int32}(C)
end

function skewlap3d_demo(N::Int)
    N ≥ 3 || throw(ArgumentError("N must be ≥ 3"))

    # Sparse identity I_{N-1}.
    In = spdiagm(0 => ones(N-1))

    # D_N = N^2 * toeplitz([-2, 1.5, 0, …], [-2, 0.5, 0, …])
    # Main diagonal -2, subdiagonal 1.5, superdiagonal 0.5: a nonsymmetric tridiagonal.
    D  = spdiagm(-1 => fill(1.5, N-2),
                  0 => fill(-2.0, N-1),
                  1 => fill(0.5, N-2))
    D  = (N^2) * D

    # A = kron(I, kron(I, D)) + kron(I, kron(D, I)) + kron(D, kron(I, I))
    A = kron(In, kron(In, D)) +
        kron(In, kron(D,  In)) +
        kron(D,  kron(In, In))

    return A
end

N = 25

C = ComplexF64.(skewlap3d_demo(N))
# σ, _, _, _ = svdsolve(C, 1, :LR)

C = to_int32_index_sparse(C)

# normC = σ[1]
# println("normC (KrylovKit.svdsolve) = ", normC)
normC = 7471.59

x_st, x_ed = -550.0, 100.0
y_st, y_ed = -200.0, 200.0
n_x, n_y = 200, 200
k = 20
