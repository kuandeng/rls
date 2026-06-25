module RLSMatrices

using LinearAlgebra
using Bumper

import Base: size, adjoint, *, show
import LinearAlgebra: mul!, ldiv!

const FloatOrComplex = Union{AbstractFloat, Complex{<:AbstractFloat}}

# ---------------------------
# Utilities
# ---------------------------
@inline function _check_mul_dims(Y::AbstractMatrix, A::AbstractMatrix, X::AbstractMatrix)
    _, n = size(A)
    k = size(X, 2)
    @assert size(X, 1) == n && size(Y) == (size(A, 1), k)
    return nothing
end

# ---------------------------
# C + sI (Shifted Matrix)
# ---------------------------
mutable struct ShiftedMat{T<:FloatOrComplex, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    C::M
    s::T
end

size(A::ShiftedMat) = size(A.C)
size(A::ShiftedMat, d::Int) = size(A.C, d)

function *(A::ShiftedMat{T}, X::AbstractMatrix{T}) where {T}
    Y = similar(X, size(A, 1), size(X, 2))
    mul!(Y, A, X)
    return Y
end

function *(A::ShiftedMat{T}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = similar(x, size(A, 1))
    mul!(reshape(Y, :, 1), A, X)
    return Y
end

function mul!(Y::AbstractMatrix{T}, A::ShiftedMat{T}, X::AbstractMatrix{T}) where {T}
    _check_mul_dims(Y, A, X)
    mul!(Y, A.C, X)

    m, n = size(A)
    n_ = min(m, n)
    @views @. Y[1:n_, :] += A.s * X[1:n_, :]
    return Y
end

function mul!(y::AbstractVector{T}, A::ShiftedMat{T}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = reshape(y, :, 1)
    mul!(Y, A, X)
    return y
end

function adjoint(A::ShiftedMat{T, M}) where {T, M}
    return ShiftedMat{T, typeof(adjoint(A.C))}(adjoint(A.C), conj(A.s))
end

function show(io::IO, A::ShiftedMat{T}) where {T}
    m, n = size(A)
    print(io, "$(m)×$(n) ShiftedMat{$T} with shift = $(A.s)")
end

function show(io::IO, ::MIME"text/plain", A::ShiftedMat{T}) where {T}
    m, n = size(A)
    println(io, "$(m)×$(n) ShiftedMat{$T}:")
    println(io, "  shift = $(A.s)")
    print(io, "  C: $(typeof(A.C))")
end

# ---------------------------
# Lazy matrix chain: A1 * A2 * ... * Al
# ---------------------------
struct MulChainMat{T<:FloatOrComplex, M<:Tuple{Vararg{AbstractMatrix{T}}}} <: AbstractMatrix{T}
    factors::M
end

function MulChainMat(A::AbstractMatrix{T}, Bs::AbstractMatrix{T}...) where {T<:FloatOrComplex}
    _check_mul_chain((A, Bs...))
    return MulChainMat{T, typeof((A, Bs...))}((A, Bs...))
end

# ------------- Type-stable last element helper -------------
@inline _last(t::Tuple{A}) where {A} = t[1]
@inline _last(t::Tuple) = _last(Base.tail(t))

# ------------- Dimension checks (chain) -------------
@inline _check_links(::Tuple) = nothing
@inline _check_links(::Tuple{A}) where {A<:AbstractMatrix} = nothing
@inline function _check_links(t::Tuple{A,B,Vararg{Any}}) where {A<:AbstractMatrix,B<:AbstractMatrix}
    @assert size(t[1], 2) == size(t[2], 1)
    _check_links(Base.tail(t))
end

function _check_mul_chain(factors::Tuple)
    @assert length(factors) ≥ 1
    _check_links(factors)
    return nothing
end


Base.size(MA::MulChainMat) = (size(MA.factors[1], 1), size(_last(MA.factors), 2))

function adjoint(C::MulChainMat)
    return MulChainMat(map(adjoint, reverse(C.factors))...)
end

function show(io::IO, MA::MulChainMat{T}) where {T}
    m, n = size(MA)
    L = length(MA.factors)
    print(io, "$(m)×$(n) MulChainMat{$T} with $(L) factors")
end

function show(io::IO, ::MIME"text/plain", MA::MulChainMat{T}) where {T}
    m, n = size(MA)
    L = length(MA.factors)
    println(io, "$(m)×$(n) MulChainMat{$T} with $(L) factors:")
    for (i, f) in enumerate(MA.factors)
        println(io, "  [$i] $(size(f, 1))×$(size(f, 2)) $(typeof(f))")
    end
end

function *(MA::MulChainMat{T}, X::AbstractMatrix{T}) where {T}
    Y = similar(X, size(MA, 1), size(X, 2))
    mul!(Y, MA, X)
    return Y
end

function *(MA::MulChainMat{T}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = similar(x, size(MA, 1))
    mul!(reshape(Y, :, 1), MA, X)
    return Y
end

function mul!(Y::AbstractMatrix{T}, MA::MulChainMat{T}, X::AbstractMatrix{T}) where {T}
    factors = MA.factors
    L = length(factors)
    _check_mul_dims(Y, MA, X)

    if L == 1
        mul!(Y, factors[1], X)
        return Y
    end

    k = size(X, 2)
    max_rows = L > 1 ? maximum(size(f, 1) for f in factors[2:end]) : size(factors[1], 1)

    @no_escape begin
        buf1 = @alloc(T, max_rows, k)
        buf2 = @alloc(T, max_rows, k)

        # tmp = A_L * X
        rows = size(factors[end], 1)
        @views mul!(buf1[1:rows, 1:k], factors[end], X)
        cur_is_buf1 = true

        # tmp = A_{L-1} * tmp ... A_1 * tmp
        for i = (L-1):-1:2
            rows = size(factors[i], 1)
            if cur_is_buf1
                @views mul!(buf2[1:rows, 1:k], factors[i], buf1[1:size(factors[i+1], 1), 1:k])
            else
                @views mul!(buf1[1:rows, 1:k], factors[i], buf2[1:size(factors[i+1], 1), 1:k])
            end
            cur_is_buf1 = !cur_is_buf1
        end

        # Final multiply into Y
        if cur_is_buf1
            @views mul!(Y, factors[1], buf1[1:size(factors[2], 1), 1:k])
        else
            @views mul!(Y, factors[1], buf2[1:size(factors[2], 1), 1:k])
        end
        nothing
    end

    return Y
end

function mul!(y::AbstractVector{T}, MA::MulChainMat{T}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = reshape(y, :, 1)
    mul!(Y, MA, X)
    return y
end

# ---------------------------
# Normal matrix: N = C' * C
# ---------------------------
struct NormalMat{T<:FloatOrComplex, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    C::M
end

size(N::NormalMat) = (size(N.C, 2), size(N.C, 2))
size(N::NormalMat, d::Int) = (@assert d == 1 || d == 2; size(N.C, 2))

function *(N::NormalMat{T}, X::AbstractMatrix{T}) where {T}
    Y = similar(X, size(N, 1), size(X, 2))
    mul!(Y, N, X)
    return Y
end

function *(N::NormalMat{T}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = similar(x, size(N, 1))
    mul!(reshape(Y, :, 1), N, X)
    return Y
end

function mul!(Y::AbstractMatrix{T}, N::NormalMat{T}, X::AbstractMatrix{T}) where {T}
    _check_mul_dims(Y, N, X)
    return mul!(Y, MulChainMat(adjoint(N.C), N.C), X)
end

function mul!(y::AbstractVector{T}, N::NormalMat{T}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = reshape(y, :, 1)
    mul!(Y, N, X)
    return y
end

adjoint(N::NormalMat) = N

function show(io::IO, N::NormalMat{T}) where {T}
    n = size(N, 1)
    print(io, "$(n)×$(n) NormalMat{$T} (C'C where C is $(size(N.C, 1))×$(size(N.C, 2)))")
end

function show(io::IO, ::MIME"text/plain", N::NormalMat{T}) where {T}
    n = size(N, 1)
    println(io, "$(n)×$(n) NormalMat{$T}:")
    print(io, "  C: $(size(N.C, 1))×$(size(N.C, 2)) $(typeof(N.C))")
end

# ---------------------------
# Identity matrix
# ---------------------------
struct IdentityMat{T<:FloatOrComplex} <: AbstractMatrix{T}
    n::Int
    function IdentityMat{T}(n::Int) where {T<:FloatOrComplex}
        n >= 0 || throw(DimensionMismatch("IdentityMat size must be nonnegative, got n = $n"))
        return new{T}(n)
    end
end

size(I::IdentityMat) = (I.n, I.n)
size(I::IdentityMat, d::Int) = (@assert d == 1 || d == 2; I.n)

function mul!(Y::AbstractMatrix{T}, I::IdentityMat{T}, X::AbstractMatrix{T}) where {T}
    _check_mul_dims(Y, I, X)
    copyto!(Y, X)
    return Y
end

function mul!(y::AbstractVector{T}, I::IdentityMat{T}, x::AbstractVector{T}) where {T}
    @assert length(x) == I.n && length(y) == I.n
    copyto!(y, x)
    return y
end

function mul!(Y::AbstractMatrix{T}, X::AbstractMatrix{T}, I::IdentityMat{T}) where {T}
    @assert size(X, 2) == I.n && size(Y) == size(X)
    copyto!(Y, X)
    return Y
end

function *(I::IdentityMat{T}, X::AbstractMatrix{T}) where {T}
    Y = similar(X, size(I, 1), size(X, 2))
    mul!(Y, I, X)
    return Y
end

function *(I::IdentityMat{T}, x::AbstractVector{T}) where {T}
    y = similar(x, size(I, 1))
    mul!(y, I, x)
    return y
end

function *(X::AbstractMatrix{T}, I::IdentityMat{T}) where {T}
    Y = similar(X, size(X, 1), size(I, 2))
    mul!(Y, X, I)
    return Y
end

@inline function ldiv!(I::IdentityMat{T}, X::AbstractMatrix{T}) where {T}
    size(X, 1) == I.n || throw(DimensionMismatch("ldiv! size mismatch for IdentityMat"))
    return X
end

@inline function ldiv!(I::IdentityMat{T}, x::AbstractVector{T}) where {T}
    length(x) == I.n || throw(DimensionMismatch("ldiv! size mismatch for IdentityMat"))
    return x
end

@inline function ldiv!(Y::AbstractMatrix{T}, I::IdentityMat{T}, X::AbstractMatrix{T}) where {T}
    size(X, 1) == I.n || throw(DimensionMismatch("ldiv! size mismatch for IdentityMat"))
    size(Y) == size(X) || throw(DimensionMismatch("ldiv! output size mismatch for IdentityMat"))
    copyto!(Y, X)
    return Y
end

@inline function ldiv!(y::AbstractVector{T}, I::IdentityMat{T}, x::AbstractVector{T}) where {T}
    length(x) == I.n || throw(DimensionMismatch("ldiv! size mismatch for IdentityMat"))
    length(y) == length(x) || throw(DimensionMismatch("ldiv! output size mismatch for IdentityMat"))
    copyto!(y, x)
    return y
end

adjoint(I::IdentityMat) = I

function show(io::IO, I::IdentityMat{T}) where {T}
    print(io, "$(I.n)×$(I.n) IdentityMat{$T}")
end

function show(io::IO, ::MIME"text/plain", I::IdentityMat{T}) where {T}
    print(io, "$(I.n)×$(I.n) IdentityMat{$T}")
end

# ---------------------------
# Permutation matrix
# ---------------------------
mutable struct PermMat{T<:FloatOrComplex, IT<:Integer, V<:AbstractVector{IT}} <: AbstractMatrix{T}
    p::V
    invp::V
    n::Int
end

function PermMat{T}(p::V, invp::V, n::Int) where {T<:FloatOrComplex, IT<:Integer, V<:AbstractVector{IT}}
    return PermMat{T, IT, V}(p, invp, n)
end

size(P::PermMat) = (P.n, P.n)
size(P::PermMat, d::Int) = (@assert d == 1 || d == 2; P.n)

function adjoint(P::PermMat{T}) where {T}
    return PermMat{T}(P.invp, P.p, P.n)
end

function show(io::IO, P::PermMat{T}) where {T}
    print(io, "$(P.n)×$(P.n) PermMat{$T}")
end

function show(io::IO, ::MIME"text/plain", P::PermMat{T}) where {T}
    println(io, "$(P.n)×$(P.n) PermMat{$T}:")
    print(io, "  p = $(P.p)")
end

function mul!(Y::AbstractMatrix{T}, P::PermMat{T}, X::AbstractMatrix{T}) where {T}
    m, n = size(X)
    invp = P.invp
    @inbounds for j = 1:n
        for i = 1:m
            Y[i, j] = X[invp[i], j]
        end
    end
    return Y
end

function mul!(y::AbstractVector{T}, P::PermMat{T}, x::AbstractVector{T}) where {T}
    invp = P.invp
    @inbounds for i = 1:length(x)
        y[i] = x[invp[i]]
    end
    return y
end

function *(P::PermMat{T}, X::AbstractMatrix{T}) where {T}
    Y = similar(X, size(P, 1), size(X, 2))
    mul!(Y, P, X)
    return Y
end

function *(P::PermMat{T}, x::AbstractVector{T}) where {T}
    y = similar(x, size(P, 1))
    mul!(y, P, x)
    return y
end

# ---------------------------
# Inverse matrix wrapper
# ---------------------------
mutable struct InvMat{T<:FloatOrComplex, M<:Union{AbstractMatrix{T}, Factorization{T}}} <: AbstractMatrix{T}
    A::M
    function InvMat(A::M) where {T<:FloatOrComplex, M<:AbstractMatrix{T}}
        size(A, 1) == size(A, 2) || throw(DimensionMismatch("InvMat only accepts square matrices, got $(size(A))"))
        new{T, M}(A)
    end
    function InvMat(A::F) where {T<:FloatOrComplex, F<:Factorization{T}}
        size(A, 1) == size(A, 2) || throw(DimensionMismatch("InvMat only accepts square matrices, got $(size(A))"))
        new{T, F}(A)
    end
end

size(Ainv::InvMat) = size(Ainv.A)
size(Ainv::InvMat, d::Int) = (@assert d == 1 || d == 2; size(Ainv.A, d))

function mul!(Y::AbstractMatrix{T}, Ainv::InvMat{T}, X::AbstractMatrix{T}) where {T}
    _check_mul_dims(Y, Ainv, X)
    ldiv!(Y, Ainv.A, X)
    return Y
end

function *(Ainv::InvMat{T}, X::AbstractMatrix{T}) where {T}
    Y = similar(X, size(Ainv, 1), size(X, 2))
    mul!(Y, Ainv, X)
    return Y
end

function *(Ainv::InvMat{T}, X::AbstractVector{T}) where {T}
    Y = similar(X)
    ldiv!(Y, Ainv.A, X)
    return Y
end

function adjoint(Ainv::InvMat{T}) where {T}
    return InvMat(adjoint(Ainv.A))
end

function show(io::IO, Ainv::InvMat{T}) where {T}
    n = size(Ainv, 1)
    print(io, "$(n)×$(n) InvMat{$T}")
end

function show(io::IO, ::MIME"text/plain", Ainv::InvMat{T}) where {T}
    n = size(Ainv, 1)
    println(io, "$(n)×$(n) InvMat{$T}:")
    print(io, "  A: $(typeof(Ainv.A))")
end

# ---------------------------
# Filled Matrix: F = [A; 0] (m×n, A is n×n or n×n special)
# ---------------------------
struct FilledMat{T<:FloatOrComplex, M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    A::M       # top block (n×n)
    m::Int     # total rows (m ≥ n)
end

size(F::FilledMat) = (F.m, size(F.A, 2))
size(F::FilledMat, d::Int) = d == 1 ? F.m : size(F.A, d)

function *(F::FilledMat{T}, X::AbstractMatrix{T}) where {T}
    Y = similar(X, size(F, 1), size(X, 2))
    mul!(Y, F, X)
    return Y
end

function *(F::FilledMat{T}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = similar(x, size(F, 1))
    mul!(reshape(Y, :, 1), F, X)
    return Y
end

# F * X = [A*X; 0]
function mul!(Y::AbstractMatrix{T}, F::FilledMat{T}, X::AbstractMatrix{T}) where {T}
    _check_mul_dims(Y, F, X)
    n_a = size(F.A, 1)
    @views mul!(Y[1:n_a, :], F.A, X)
    if F.m > n_a
        @views Y[n_a+1:F.m, :] .= zero(T)
    end
    return Y
end

function mul!(y::AbstractVector{T}, F::FilledMat{T}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = reshape(y, :, 1)
    mul!(Y, F, X)
    return y
end

# F' * X = [A', 0] * X = A' * X[1:n_a, :]
function mul!(Y::AbstractMatrix{T}, Fa::Adjoint{T, <:FilledMat{T}}, X::AbstractMatrix{T}) where {T}
    F = parent(Fa)
    n_a = size(F.A, 1)
    @views mul!(Y, adjoint(F.A), X[1:n_a, :])
    return Y
end

function mul!(y::AbstractVector{T}, Fa::Adjoint{T, <:FilledMat{T}}, x::AbstractVector{T}) where {T}
    X = reshape(x, :, 1)
    Y = reshape(y, :, 1)
    mul!(Y, Fa, X)
    return y
end

function *(Fa::Adjoint{T, <:FilledMat{T}}, X::AbstractMatrix{T}) where {T}
    F = parent(Fa)
    n = size(F.A, 2)
    Y = similar(X, n, size(X, 2))
    mul!(Y, Fa, X)
    return Y
end

function *(Fa::Adjoint{T, <:FilledMat{T}}, x::AbstractVector{T}) where {T}
    F = parent(Fa)
    n = size(F.A, 2)
    y = similar(x, n)
    mul!(reshape(y, :, 1), Fa, reshape(x, :, 1))
    return y
end

function show(io::IO, F::FilledMat{T}) where {T}
    m, n = size(F)
    print(io, "$(m)×$(n) FilledMat{$T}")
end

function show(io::IO, ::MIME"text/plain", F::FilledMat{T}) where {T}
    m, n = size(F)
    n_a = size(F.A, 1)
    println(io, "$(m)×$(n) FilledMat{$T}:")
    print(io, "  A: $(n_a)×$(size(F.A, 2)) $(typeof(F.A))")
end

export ShiftedMat, MulChainMat, NormalMat, IdentityMat, PermMat, InvMat, FilledMat

end # module RLSMatrices
