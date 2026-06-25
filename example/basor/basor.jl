
include("../../src/RLS.jl")
using KrylovKit
using ToeplitzMatrices

function basor_mat(N::Int)
    r = -im ./ collect(1:N)
    c = vcat(-im, π, im ./ collect(1:N-2))
    T = Toeplitz(c, r)
    return T
end

function basor_gram_fast(N::Int)
    H = zeros(Float64, N)
    H2 = zeros(Float64, N)

    @inbounds for n in 1:N
        H[n] = (n == 1 ? 0.0 : H[n - 1]) + 1.0 / n
        H2[n] = (n == 1 ? 0.0 : H2[n - 1]) + 1.0 / (n * n)
    end

    Hn(n::Int) = n <= 0 ? 0.0 : H[n]
    H2n(n::Int) = n <= 0 ? 0.0 : H2[n]

    G = Matrix{ComplexF64}(undef, N, N)

    @inbounds for j in 1:N
        G[j, j] = H2n(j) + (j < N ? π^2 : 0.0) + H2n(N - j - 1)

        for k in j+1:N
            d = k - j
            s1 = (Hn(j) + Hn(d) - Hn(k)) / d
            s2 = 2.0 * Hn(d - 1) / d
            tail_len = N - k - 1
            s3 = tail_len > 0 ? (Hn(tail_len) + Hn(d) - Hn(tail_len + d)) / d : 0.0
            sim = -π * (1.0 + (k < N ? 1.0 : 0.0)) / d

            G[j, k] = complex(s1 - s2 + s3, sim)
            G[k, j] = conj(G[j, k])
        end
    end

    return Hermitian(G)
end

C = basor_mat(2000)
# CTC = basor_gram_fast(100)
x_st, x_ed = -4, 6.5
y_st, y_ed = -5, 2
n_x, n_y = 200, 200
# k = 10
# rec = recursion_grid(x_st, x_ed, y_st, y_ed, n_x, n_y, k)