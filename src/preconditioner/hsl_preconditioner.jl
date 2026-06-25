

# Constants and error codes.
const SF_DEFAULT  = 2.0
const SF2_DEFAULT = 4.0
const ALPHAM      = 1.0e-3

const MI28_ERROR_ALLOCATION   = -1
const MI28_ERROR_ROW_TOO_SMALL= -2
const MI28_ERROR_VAL_TOO_SMALL= -3
const MI28_ERROR_N_OOR        = -4
const MI28_ERROR_PTR          = -5
const MI28_ERROR_MISS_DIAG    = -6
const MI28_ERROR_MC77         = -7
const MI28_ERROR_MC64         = -8
const MI28_ERROR_SINGULAR     = -9
const MI28_ERROR_SCALE        = -10
const MI28_ERROR_USER_PERM    = -11
const MI28_ERROR_MC61         = -12
const MI28_ERROR_MC68         = -13
const MI28_ERROR_DEALLOCATION = -14

const MI28_WARNING_OOR_IDX    = 1
const MI28_WARNING_DUP_IDX    = 2
const MI28_WARNING_MC64       = 3
const MI28_WARNING_MC61       = 4
const MI28_WARNING_NEG_DIAG   = 5

# Data structures.
Base.@kwdef mutable struct mi35_keep{T<:FloatOrComplex, RT<:AbstractFloat, IT<:Integer}
    fact_ptr::AbstractVector{IT} = IT[]
    fact_row::AbstractVector{IT} = IT[]
    fact_val::AbstractVector{T} = T[]
    scale::AbstractVector{RT} = RT[]
    invp::AbstractVector{IT} = IT[]
    perm::AbstractVector{IT} = IT[]
    w::AbstractVector{T} = T[]
end

mi35_keep{T, RT}() where {T<:FloatOrComplex, RT<:AbstractFloat} = mi35_keep{T, RT, Int}()
const mi28_keep = mi35_keep

Base.@kwdef mutable struct mi35_control{RT<:AbstractFloat}
    alpha::RT = zero(RT)
    check::Bool = true
    iorder::Int = 6
    iscale::Int = 1
    lowalpha::RT = RT(ALPHAM)
    maxshift::Int = 3
    rrt::Bool = false
    shift_factor::RT = RT(SF_DEFAULT)
    shift_factor2::RT = RT(SF2_DEFAULT)
    small::RT = RT(1e-20)
    tau1::RT = RT(1e-3)
    tau2::RT = RT(1e-4)
    unit_error::Int = 6
    unit_warning::Int = 6
end
const mi28_control = mi35_control

@inline function _normalize_mi35_shift_control(control::mi35_control{RT}) where {RT<:AbstractFloat}
    alpha_in = max(zero(RT), control.alpha)
    lowalpha = control.lowalpha > zero(RT) ? control.lowalpha : RT(ALPHAM)
    if alpha_in > zero(RT)
        lowalpha = alpha_in
    end
    shift_factor = control.shift_factor > one(RT) ? control.shift_factor : RT(SF_DEFAULT)
    shift_factor2 = control.shift_factor2 > one(RT) ? control.shift_factor2 : RT(SF2_DEFAULT)
    return (; alpha_in, lowalpha, shift_factor, shift_factor2)
end

include("permute_matrix.jl")
include("scale.jl")
include("ictkl3.jl")
include("ictkl3_new.jl")
include("ichsd.jl")

function mi28_factorize!(
    n::Int,
    ptr::AbstractVector{IT},
    row::AbstractVector{IT},
    val::AbstractVector{T},
    lsize::Int,
    rsize::Int,
    keep::mi35_keep{T, RT, IK},
    control::mi35_control{RT};
    out_index_type = Int
) where {T<:FloatOrComplex, RT<:AbstractFloat, IT<:Integer, IK<:Integer}
    (out_index_type isa Type && out_index_type <: Integer) || throw(ArgumentError("out_index_type must be <: Integer, got $out_index_type"))

    ptrl = nothing
    rowl = nothing
    vall = nothing

    if keep.perm !== nothing && !isempty(keep.perm)
        @no_escape begin
            nnz = Int(ptr[end] - one(IT))
            ptrp = @alloc(IT, n + 1)
            rowp = @alloc(IT, nnz)
            valp = @alloc(T, nnz)
            work = @alloc(IT, n)

            if IK !== IT
                invp_it = @alloc(IT, n)
                @inbounds for i in 1:n
                    invp_it[i] = IT(keep.invp[i])
                end
                permute_lower_csc!(n, ptr, row, val, invp_it, ptrp, rowp, valp, work)
            else
                permute_lower_csc!(n, ptr, row, val, keep.invp, ptrp, rowp, valp, work)
            end

            if control.iscale > 0
                keep.scale = compute_scaling(n, ptrp, rowp, valp, control.iscale)
            else
                keep.scale = ones(RT, n)
            end
            ptrl, rowl, vall, _ = ichsd(n, ptrp, rowp, valp, lsize, rsize, keep, control; out_index_type = out_index_type)
        end
    else
        if control.iscale > 0
            keep.scale = compute_scaling(n, ptr, row, val, control.iscale)
        else
            keep.scale = ones(RT, n)
        end
        ptrl, rowl, vall, _ = ichsd(n, ptr, row, val, lsize, rsize, keep, control; out_index_type = out_index_type)
    end
    return ptrl, rowl, vall
end

# Compatibility shim for old call sites that still pass `info`.
function mi28_factorize!(
    n::Int,
    ptr::AbstractVector{IT},
    row::AbstractVector{IT},
    val::AbstractVector{T},
    lsize::Int,
    rsize::Int,
    keep::mi35_keep{T, RT, IK},
    control::mi35_control{RT},
    _info;
    out_index_type = Int
) where {T<:FloatOrComplex, RT<:AbstractFloat, IT<:Integer, IK<:Integer}
    return mi28_factorize!(n, ptr, row, val, lsize, rsize, keep, control; out_index_type = out_index_type)
end

@inline function _build_hsl_normal_eq(
    C::AbstractMatrix{T},
    z::T
) where {T<:FloatOrComplex}
    Cz = C - z * I
    if Cz isa SparseMatrixCSC
        CzH = sparse(adjoint(Cz))
        return sparse(LowerTriangular(SparseArrays.spmatmul(CzH, Cz)))
    end
    return sparse(LowerTriangular(adjoint(Cz) * Cz))
end

function get_hsl_LLTP(
    C::AbstractMatrix{T},
    z::T,
    tau1::RT,
    tau2::RT,
    α::RT,
    lsize::Int,
    rsize::Int,
    p::AbstractVector{IT};
    out_index_type::Union{Nothing, Type{<:Integer}} = nothing,
    NC::Union{Nothing, AbstractMatrix{T}} = nothing
) where {T<:FloatOrComplex, RT<:AbstractFloat, IT<:Integer}
    m, n = size(C)
    m == n || throw(DimensionMismatch("get_hsl_LLTP only supports square C, got size $(size(C))"))
    length(p) == n || throw(DimensionMismatch("permutation length mismatch: length(p)=$(length(p)) n=$n"))

    NC_local = if NC === nothing
        _build_hsl_normal_eq(C, z)
    else
        size(NC, 1) == n && size(NC, 2) == n ||
            throw(DimensionMismatch("NC size mismatch: expected ($n, $n), got $(size(NC))"))
        NC_sparse = NC isa SparseMatrixCSC ? NC : sparse(NC)
        sparse(LowerTriangular(NC_sparse))
    end

    OT = out_index_type === nothing ? (NC === nothing ? (C isa SparseMatrixCSC ? eltype(C.colptr) : IT) : eltype(NC_local.colptr)) : out_index_type
    OT <: Integer || throw(ArgumentError("out_index_type must be <: Integer, got $OT"))
    n <= typemax(OT) || throw(ArgumentError("matrix size n=$n exceeds typemax($OT)=$(typemax(OT))"))

    ptr, row, val = NC_local.colptr, NC_local.rowval, NC_local.nzval
    PT = eltype(ptr)
    n <= typemax(PT) || throw(ArgumentError("matrix size n=$n exceeds typemax($PT)=$(typemax(PT))"))

    ptrl = nothing
    rowl = nothing
    vall = nothing
    p_out = Vector{OT}(undef, n)
    invp_out = Vector{OT}(undef, n)
    keep_scale = Vector{RT}(undef, n)

    @no_escape begin
        p_perm = @alloc(PT, n)
        invp_perm = @alloc(PT, n)

        @inbounds for i in 1:n
            p_perm[i] = PT(p[i])
        end
        @inbounds for i in 1:n
            invp_perm[Int(p_perm[i])] = PT(i)
        end

        keep = mi35_keep{T, RT, PT}()
        keep.perm = p_perm
        keep.invp = invp_perm

        control = mi35_control{RT}()
        control.tau1 = tau1
        control.tau2 = tau2
        control.alpha = α
        control.rrt = false

        ptrl, rowl, vall = mi28_factorize!(n, ptr, row, val, lsize, rsize, keep, control; out_index_type = OT)
        copyto!(keep_scale, keep.scale)

        @inbounds for i in 1:n
            p_out[i] = OT(p_perm[i])
            invp_out[i] = OT(invp_perm[i])
        end
    end

    nL = Int(ptrl[end] - one(eltype(ptrl)))
    if OT === eltype(ptrl)
        L = SparseMatrixCSC(n, n, ptrl, rowl[1:nL], vall[1:nL])
    else
        ptrl_ot = OT.(ptrl)
        rowl_ot = OT.(view(rowl, 1:nL))
        L = SparseMatrixCSC(n, n, ptrl_ot, rowl_ot, vall[1:nL])
    end

    @inbounds for i in 1:n
        L[i, i] = 1 / L[i, i]
    end
    # Apply left diagonal scaling in-place to keep index type stable.
    @inbounds for col in 1:n
        for k in L.colptr[col]:(L.colptr[col+1]-one(eltype(L.colptr)))
            L.nzval[k] *= (one(RT) / keep_scale[Int(L.rowval[k])])
        end
    end
    if eltype(L.colptr) !== OT
        n <= typemax(OT) || throw(ArgumentError("matrix size n=$n exceeds typemax($OT)=$(typemax(OT))"))
        L = SparseMatrixCSC{eltype(L.nzval), OT}(L)
    end
    return SparseLLTP(L, p_out, invp_out)
end
