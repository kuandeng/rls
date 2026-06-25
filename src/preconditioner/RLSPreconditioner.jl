using LinearAlgebra
using SparseArrays
using Random
using Bumper
import LinearAlgebra: ldiv!


mutable struct SparseLLTP{T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}, MF<:AbstractMatrix{T}} <: AbstractMatrix{T}
    L::LowerTriangular{T, SparseMatrixCSC{T, IT}}
    p::VP
    invp::VP
    invF::MF
end

SparseLLTP(
    L::SparseMatrixCSC{T, IT},
    p::VP,
    invp::VP
) where {T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}} = begin
    Lt = LowerTriangular(L)
    P = PermMat{T}(p, invp, size(L, 2))
    invLT = InvMat(adjoint(Lt))
    invF = RLSMatrices.MulChainMat(P, invLT)
    return SparseLLTP{T, IT, VP, typeof(invF)}(Lt, p, invp, invF)
end

SparseLLTP{T, IT}() where {T<:FloatOrComplex, IT<:Integer} = SparseLLTP(
    SparseMatrixCSC{T, IT}(0, 0, IT[one(IT)], IT[], T[]),
    IT[],
    IT[]
)

SparseLLTP{T}() where {T<:FloatOrComplex} = SparseLLTP{T, Int}()

Base.size(K::SparseLLTP{T, IT, VP}) where {T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}} = size(K.L)
Base.size(K::SparseLLTP{T, IT, VP}, d::Int) where {T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}} = size(K.L, d)

function mul_coupled_ms!(
    MV::AbstractMatrix{T},
    SV::AbstractMatrix{T},
    M::AbstractMatrix{T},
    S::AbstractMatrix{T},
    V::AbstractMatrix{T}
) where {T<:FloatOrComplex}
    mul!(MV, M, V)
    mul!(SV, S, V)
    return nothing
end

function mul_coupled_ms!(
    MV::AbstractMatrix{T},
    SV::AbstractMatrix{T},
    M::RLSMatrices.MulChainMat{T},
    S::RLSMatrices.MulChainMat{T},
    V::AbstractMatrix{T}
) where {T<:FloatOrComplex}
    Mf = M.factors
    Sf = S.factors

    # Fast path: M = M0 * F and S = S0 * F, compute F*V only once.
    if length(Mf) == 2 && length(Sf) == 2 && Mf[2] === Sf[2]
        F = Mf[2]
        k = size(V, 2)
        @no_escape begin
            FV = @alloc(T, size(F, 1), k)
            mul!(FV, F, V)
            mul!(MV, Mf[1], FV)
            mul!(SV, Sf[1], FV)
            nothing
        end
    else
        mul!(MV, M, V)
        mul!(SV, S, V)
    end
    return nothing
end


function ldiv!(
    X::AbstractMatrix{T},
    K::SparseLLTP{T, IT, VP},
    B::AbstractMatrix{T}
) where {T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}}
    n = size(K, 1)
    k = size(B, 2)
    @assert size(B, 1) == n
    @assert size(X) == (n, k)

    p = K.p
    invp = K.invp
    Lt = adjoint(K.L)

    @no_escape begin
        temp = @alloc(T, n, k)

        # temp = P' * B
        @inbounds for j = 1:k
            for i = 1:n
                temp[i, j] = B[p[i], j]
            end
        end

        # X = L \ temp
        ldiv!(X, K.L, temp)
        # temp = L' \ X
        ldiv!(temp, Lt, X)

        # X = P * temp
        @inbounds for j = 1:k
            for i = 1:n
                X[i, j] = temp[invp[i], j]
            end
        end
    end
    return X
end

ldiv!(K::SparseLLTP{T, IT, VP}, B::AbstractMatrix{T}) where {T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}} = ldiv!(B, K, B)


function ldiv!(
    x::AbstractVector{T},
    K::SparseLLTP{T, IT, VP},
    b::AbstractVector{T}
) where {T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}}
    n = size(K, 1)
    @assert length(b) == n
    @assert length(x) == n

    p = K.p
    invp = K.invp
    Lt = adjoint(K.L)

    @no_escape begin
        temp = @alloc(T, n)

        # temp = P' * b
        @inbounds for i = 1:n
            temp[i] = b[p[i]]
        end

        # x = L \ temp
        ldiv!(x, K.L, temp)
        # temp = L' \ x
        ldiv!(temp, Lt, x)

        # x = P * temp
        @inbounds for i = 1:n
            x[i] = temp[invp[i]]
        end
    end
    return x
end

ldiv!(K::SparseLLTP{T, IT, VP}, b::AbstractVector{T}) where {T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}} = ldiv!(b, K, b)





mutable struct SparseLLTP_AD{
    T<:FloatOrComplex,
    IT<:Integer,
    VP<:AbstractVector{IT},
    MK<:SparseLLTP{T, IT, VP},
    ME<:AbstractMatrix{T},
    MG<:AbstractMatrix{T},
    FF<:Factorization{T}
} <: AbstractMatrix{T}
    K::MK
    E::ME
    GramE::MG
    F_GramE::FF
end

function SparseLLTP_AD(
    K::SparseLLTP{T, IT, VP},
    E::ME,
    GramE::MG
) where {T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT}, ME<:AbstractMatrix{T}, MG<:AbstractMatrix{T}}
    F_GramE = rls_cholesky!(copy(GramE))
    return SparseLLTP_AD{T, IT, VP, typeof(K), ME, MG, typeof(F_GramE)}(K, E, GramE, F_GramE)
end

function SparseLLTP_AD(
    K::SparseLLTP{T, IT, VP},
    E::ME,
    GramE::MG,
    F_GramE::FF
) where {
    T<:FloatOrComplex, IT<:Integer, VP<:AbstractVector{IT},
    ME<:AbstractMatrix{T}, MG<:AbstractMatrix{T}, FF<:Factorization{T}
}
    return SparseLLTP_AD{T, IT, VP, typeof(K), ME, MG, FF}(K, E, GramE, F_GramE)
end

SparseLLTP_AD{T, IT}() where {T<:FloatOrComplex, IT<:Integer} = SparseLLTP_AD(
    SparseLLTP{T, IT}(),
    zeros(T, 0, 0),
    Matrix{T}(I, 0, 0)
)

SparseLLTP_AD{T}() where {T<:FloatOrComplex} = SparseLLTP_AD{T, Int}()

Base.size(Kad::SparseLLTP_AD{T}) where {T<:FloatOrComplex} = size(Kad.K)
Base.size(Kad::SparseLLTP_AD{T}, d::Int) where {T<:FloatOrComplex} = size(Kad.K, d)

function update_sparse_lldiv_ad_basis!(
    Kad::SparseLLTP_AD{T},
    X::AbstractMatrix{T},
    CX::AbstractMatrix{T}
) where {T<:FloatOrComplex}
    n, ks = size(Kad.E)
    @assert size(X) == (n, ks)

    copyto!(Kad.E, X)

    mul!(Kad.GramE, CX', CX)
    @no_escape begin
        GramE_tmp = @alloc(T, ks, ks)
        copyto!(GramE_tmp, Kad.GramE)
        rls_cholesky!(GramE_tmp)
        copyto!(Kad.F_GramE.factors, GramE_tmp)
    end
    return nothing
end


function ldiv!(
    X::AbstractMatrix{T},
    Kad::SparseLLTP_AD{T},
    B::AbstractMatrix{T}
) where {T<:FloatOrComplex}
    n = size(Kad, 1)
    k = size(B, 2)
    ks = size(Kad.E, 2)
    @assert size(B, 1) == n
    @assert size(X) == (n, k)
    @assert size(Kad.E, 1) == n
    @assert size(Kad.GramE) == (ks, ks)

    @no_escape begin
        proj = @alloc(T, n, k)
        coeff = @alloc(T, ks, k)

        # proj = E * (GramE \ (E' * B))
        mul!(coeff, Kad.E', B)
        ldiv!(Kad.F_GramE, coeff)
        mul!(proj, Kad.E, coeff)

        # X = K^{-1} * B
        ldiv!(X, Kad.K, B)
        # X += proj
        @views X .+= proj
    end
    return X
end

ldiv!(Kad::SparseLLTP_AD{T}, B::AbstractMatrix{T}) where {T<:FloatOrComplex} = ldiv!(B, Kad, B)

function ldiv!(
    x::AbstractVector{T},
    Kad::SparseLLTP_AD{T},
    b::AbstractVector{T}
) where {T<:FloatOrComplex}
    n = size(Kad, 1)
    ks = size(Kad.E, 2)
    @assert length(b) == n
    @assert length(x) == n
    @assert size(Kad.E, 1) == n
    @assert size(Kad.GramE) == (ks, ks)

    @no_escape begin
        proj = @alloc(T, n)
        coeff = @alloc(T, ks)

        # proj = E * (GramE \ (E' * b))
        mul!(coeff, Kad.E', b)
        ldiv!(Kad.F_GramE, coeff)
        mul!(proj, Kad.E, coeff)

        # x = K^{-1} * b
        ldiv!(x, Kad.K, b)
        # x += proj
        @views x .+= proj
    end
    return x
end

ldiv!(Kad::SparseLLTP_AD{T}, b::AbstractVector{T}) where {T<:FloatOrComplex} = ldiv!(b, Kad, b)

include("./hsl_preconditioner.jl")
