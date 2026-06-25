# recycled space for (M - zS)^*(M - zS) eigenvalue problems
# V : orthogonal basis  
# H1: (MV)^*MV 
# H2: (MV)^*SV 
# H3: (SV)^*SV

# J1: M^*MV
# J2: M^*SV
# J3: S^*MV
# J4: S^*SV
# J5: MV
# J6: SV

# Optimizations for special S ∈ C^{m×n}:
# - Use FilledMat{T} for S = [A; 0]: mul!/adjoint mul! auto-dispatch zero structure
# - sn: effective row dim of S (= size(A,1) for FilledMat, = m for general S)
# - orth_S: S^*S = I_n => H3 = I, J4 = V
# - implict_update: Bool, if false, update J3, J4 using J5, J6 with S'
# - implict_recycle: Bool, if false, only maintain J5/J6 (skip J1~J4 cache)

# Use one grid step as the default distance threshold for recycle failure.
@inline function _default_recycle_fail_dist_thresh(rec)
    dx = rec.n_x > 1 ? abs(rec.x_ed - rec.x_st) / (rec.n_x - 1) : 0.0
    dy = rec.n_y > 1 ? abs(rec.y_ed - rec.y_st) / (rec.n_y - 1) : 0.0
    return 1.01*max(dx, dy)
end

# Resolve user-provided recycle failure distance, accepting NaN/Inf as "use default".
@inline function _resolve_recycle_fail_dist_thresh(
    thresh::Real,
    rec
)
    if isfinite(thresh)
        thresh >= 0 || throw(ArgumentError("recycle_fail_dist_thresh must be nonnegative, got $thresh"))
        return Float64(thresh)
    end
    return _default_recycle_fail_dist_thresh(rec)
end

Base.@kwdef mutable struct RecycledSpace{T}
    orth_S::Bool = false
    implict_update::Bool = false
    implict_recycle::Bool = true
    p_max::Int
    p_min::Int
    p::Int
    m::Int
    n::Int
    sn::Int      # effective row dim of S (n for FilledMat, m otherwise)
    r::Int
    pts::Vector{T}
    V::Matrix{T}
    L::Matrix{T}
    H1::Matrix{T}
    H2::Matrix{T}
    H3::Matrix{T}
    J1::Matrix{T}
    J2::Matrix{T}
    J3::Matrix{T}
    J4::Matrix{T}   # Shares storage with V when orth_S is true.
    J5::Matrix{T} 
    J6::Matrix{T}
end

"""
    RecycledSpace(T, m, n, p_max, r; kwargs...)

Allocate the cache used to recycle basis blocks across shifted normal equations.
"""
function RecycledSpace(
    T::Type,
    m::Int,
    n::Int,
    p_max::Int,
    r::Int;
    sn::Int = m,
    orth_S::Bool = false,
    implict_update::Bool = false,
    implict_recycle::Bool = true,
    p_min::Int = 1
)
    p_max > 0 || throw(ArgumentError("p_max must be positive, got $p_max"))
    0 <= p_min <= p_max || throw(ArgumentError("p_min must satisfy 0 <= p_min <= p_max, got p_min=$p_min, p_max=$p_max"))

    pr = p_max * r
    V = zeros(T, n, pr)
    J1 = implict_recycle ? zeros(T, n, pr) : zeros(T, 0, 0)
    J2 = implict_recycle ? zeros(T, n, pr) : zeros(T, 0, 0)
    J3 = implict_recycle ? zeros(T, n, pr) : zeros(T, 0, 0)
    J4 = if implict_recycle
        orth_S ? V : zeros(T, n, pr)  # orth_S: J4 shares storage with V.
    else
        zeros(T, 0, 0)
    end

    RecycledSpace{T}(
        orth_S = orth_S,
        implict_update = implict_update,
        implict_recycle = implict_recycle,
        p_max = p_max,
        p_min = p_min,
        p = 0,
        m = m,
        n = n,
        sn = sn,
        r = r,
        pts = zeros(T, p_max),
        V = V,
        L = zeros(T, pr, pr),
        H1 = zeros(T, pr, pr),
        H2 = zeros(T, pr, pr),
        H3 = zeros(T, pr, pr),
        J1 = J1,
        J2 = J2,
        J3 = J3,
        J4 = J4,
        J5 = zeros(T, m, pr), 
        J6 = zeros(T, m, pr),  
    )
end

"""
    RecycledSpace(M, S; p_max, r, kwargs...)

Construct a recycled-space cache from operator dimensions and promoted element type.
"""
function RecycledSpace(
    M::AbstractMatrix{TM},
    S::AbstractMatrix{TS};
    p_max::Int,
    r::Int,
    sn::Int = size(S, 1),
    orth_S::Bool = false,
    implict_update::Bool = false,
    implict_recycle::Bool = true,
    p_min::Int = 1
) where {TM, TS}
    size(M) == size(S) || throw(DimensionMismatch("M and S must have the same size, got $(size(M)) and $(size(S))"))
    T = promote_type(TM, TS)
    m, n = size(M)
    return RecycledSpace(
        T,
        m,
        n,
        p_max,
        r;
        sn = sn,
        orth_S = orth_S,
        implict_update = implict_update,
        implict_recycle = implict_recycle,
        p_min = p_min
    )
end

"""
    updateRecycledSpace!(M, S, z, recySp, X; tau = 1.5, r_sub = 1, force_add = false)

Update the recycled basis with candidate block `X`, dropping old blocks when needed.
"""
function updateRecycledSpace!(
    M::AbstractMatrix{T},
    S::AbstractMatrix{T},
    z::T,
    recySp::RecycledSpace{T},
    X::AbstractMatrix{T};
    τ::Float64 = 1.5,
    r_sub::Int = 1,
    force_add::Bool = false
) where T
    @unpack p, m, n, r, pts, V, L, H1, H2, H3, J1, J2, J3, J4, J5, J6 = recySp
    p_max = recySp.p_max
    p_min = recySp.p_min

    @no_escape begin
        if p > 0
            @views V0 = V[:, 1:p*r]
            @views X_sub = X[:, 1:r_sub]

            # compute residual initial
            X_temp = @alloc(T, n, r_sub)
            R = @alloc(T, r_sub, r_sub)

            X_temp .= X_sub
            rls_orth_robust!(V0, X_temp, nothing, R)
            res0 = norm(R)

            l = 0
            if !force_add
                # find l by checking residual reduction, but never scan past the
                # largest deletion allowed if p_min is interpreted on the final
                # block count after the new block is added.
                l_max = min(p - 1, p + 1 - p_min)
                l = l_max
                for i = 1:(l_max + 1)
                    @views Vl = V[:, (i*r+1):p*r]
                    X_temp .= X_sub
                    rls_orth_robust!(Vl, X_temp, nothing, R)
                    resl = norm(R)
                    if resl/res0 > τ
                        l = i - 1
                        break
                    end
                end
            end

            # update the blocks
            if l > 0 
                _shift_blocks!(recySp, l)
            end 

            if l == 0 && p == p_max
                _shift_blocks!(recySp, 1)
            end
        end
        
        # add new blocks to H and J
        _add_blocks!(recySp, X, M, S, z)

        _block_unitary_update!(recySp, S)
        nothing
    end

end

# Apply block-unitary rotations so the newest block is triangularized against old blocks.
function _block_unitary_update!(recySp::RecycledSpace{T}, S::AbstractMatrix{T}) where T
    @unpack p, m, n, r, sn, V, L, H1, H2, H3, J1, J2, J3, J4, J5, J6 = recySp
    _orth_S = recySp.orth_S
    _implict_update = recySp.implict_update
    _implict_recycle = recySp.implict_recycle

    if p <= 1
        return nothing
    end
    
    @no_escape begin
        U = @alloc(T, 2r, 2r)
        
        @views for i = 1:p-1
            ri = L[(i-1)*r+1:i*r, (p-1)*r+1:p*r]
            rend = L[(p-1)*r+1:p*r, (p-1)*r+1:p*r]
            bu = rls_block_unitary_rot!(rend, ri, p, i, U)
            
            lmul!(bu, L[1:p*r, 1:p*r])
            rmul!(V[:, 1:p*r], adjoint(bu))

            rmul!(J5[:, 1:p*r], adjoint(bu))
            rmul!(J6[1:sn, 1:p*r], adjoint(bu))  # sn handles FilledMat dimensions.

            if _implict_recycle
                rmul!(J1[:, 1:p*r], adjoint(bu))
                rmul!(J2[:, 1:p*r], adjoint(bu))
                if _implict_update
                    rmul!(J3[:, 1:p*r], adjoint(bu))
                    if !_orth_S
                        rmul!(J4[:, 1:p*r], adjoint(bu))
                    end
                else
                    # update the corresponding blocks in J3 and J4 using S'
                    i1, i2 = bu.i1, bu.i2
                    idx1 = (i1-1)*r+1 : i1*r
                    idx2 = (i2-1)*r+1 : i2*r
                    @views mul!(J3[:, idx1], S', J5[:, idx1])
                    @views mul!(J3[:, idx2], S', J5[:, idx2])
                    if !_orth_S
                        @views mul!(J4[:, idx1], S', J6[:, idx1])
                        @views mul!(J4[:, idx2], S', J6[:, idx2])
                    end
                end
            end

            rmul!(H1[1:p*r, 1:p*r], adjoint(bu))
            lmul!(bu, H1[1:p*r, 1:p*r])
            rmul!(H2[1:p*r, 1:p*r], adjoint(bu))
            lmul!(bu, H2[1:p*r, 1:p*r])
            if !_orth_S
                rmul!(H3[1:p*r, 1:p*r], adjoint(bu))
                lmul!(bu, H3[1:p*r, 1:p*r])
            end
        end

        # When orth_S is true, J4 aliases V and is already updated by rotating V.
        nothing
    end

end


# Drop the first `l` recycled blocks and compact all cached matrices in place.
function _shift_blocks!(recySp::RecycledSpace{T}, l::Int) where T
    @unpack p, m, n, r, sn, pts, V, L, H1, H2, H3, J1, J2, J3, J4, J5, J6 = recySp
    _orth_S = recySp.orth_S
    _implict_recycle = recySp.implict_recycle

    src_cols = l*r+1 : p*r
    dst_cols = 1 : (p-l)*r
    
    # NOTE:
    # dst/src ranges can overlap during left-shift. Use explicit forward copy
    # (dst < src) to avoid overwrite corruption from broadcast assignment.
    @views begin
        for i in 1:(p - l)
            pts[i] = pts[i + l]
        end

        ncols = length(dst_cols)
        for j = 1:ncols
            dj = dst_cols[j]
            sj = src_cols[j]
            copyto!(view(V, :, dj), view(V, :, sj))
            if _implict_recycle
                copyto!(view(J1, :, dj), view(J1, :, sj))
                copyto!(view(J2, :, dj), view(J2, :, sj))
                copyto!(view(J3, :, dj), view(J3, :, sj))
                if !_orth_S  # When orth_S is true, moving V also moves J4.
                    copyto!(view(J4, :, dj), view(J4, :, sj))
                end
            end
            copyto!(view(J5, :, dj), view(J5, :, sj))
            copyto!(view(J6, 1:sn, dj), view(J6, 1:sn, sj))
        end

        for jj = 1:ncols
            dj = dst_cols[jj]
            sj = src_cols[jj]
            for ii = 1:ncols
                di = dst_cols[ii]
                si = src_cols[ii]
                L[di, dj] = L[si, sj]
                H1[di, dj] = H1[si, sj]
                H2[di, dj] = H2[si, sj]
                if !_orth_S
                    H3[di, dj] = H3[si, sj]
                end
            end
        end
    end
    
    recySp.p = p - l
end


# Orthogonalize and append one new recycled block, then refresh cached products.
function _add_blocks!(recySp::RecycledSpace{T}, X::AbstractMatrix{T}, 
                      M::AbstractMatrix, S::AbstractMatrix, 
                      z::T) where T
    @unpack p, m, n, r, sn, pts, V, L, H1, H2, H3, J1, J2, J3, J4, J5, J6 = recySp
    _orth_S = recySp.orth_S
    _implict_recycle = recySp.implict_recycle

    old_cols = 1:p*r
    new_cols = p*r+1:(p+1)*r

    @no_escape @views begin
        # Update V and L.
        V_new = V[:, new_cols]
        copyto!(V_new, X) 
        if p > 0
            rls_orth_robust!(V[:, old_cols], V_new, L[old_cols, new_cols], L[new_cols, new_cols])
        else 
            rls_qr!(V_new, L[new_cols, new_cols])
        end

        # Compute MV_new and SV_new.
        J5_new = J5[:, new_cols]
        J6_new = J6[:, new_cols]
        # M,S may share a right factor (e.g. invF); coupled path computes it once.
        mul_coupled_ms!(J5_new, J6_new, M, S, V_new)

        if _implict_recycle
            # Compute J1 through J4.
            J1_new = J1[:, new_cols]
            J2_new = J2[:, new_cols]
            J3_new = J3[:, new_cols]
            J4_new = J4[:, new_cols]

            mul!(J1_new, M', J5_new)
            if sn == m
                # Fast path: keep triangular structure from M' (typically hits optimized BLAS path).
                mul!(J2_new, M', J6_new)
            else
                # Avoid view-of-view on J6_new to reduce strided fallback overhead.
                mul!(J2_new, M[1:sn, :]', J6[1:sn, new_cols])
            end
            mul!(J3_new, S', J5_new)                    # FilledMat adjoint dispatch

            if !_orth_S  # When orth_S is true, J4 aliases V and follows V updates.
                mul!(J4_new, S', J6_new)
            end

            # Update diagonal H blocks using H = V' * J, O(nr^2) instead of O(mr^2).
            mul!(H1[new_cols, new_cols], V_new', J1_new)
            mul!(H2[new_cols, new_cols], V_new', J2_new)
            if _orth_S
                H3[new_cols, new_cols] .= zero(T)
                for i in 1:r
                    H3[p*r+i, p*r+i] = one(T)
                end
            else
                mul!(H3[new_cols, new_cols], V_new', J4_new)
            end

            # Update off-diagonal H blocks.
            if p > 0
                mul!(H1[old_cols, new_cols], J1[:, old_cols]', V_new)
                mul!(H2[old_cols, new_cols], J3[:, old_cols]', V_new)
                H1[new_cols, old_cols] .= H1[old_cols, new_cols]'
                mul!(H2[new_cols, old_cols], V_new', J2[:, old_cols])

                if _orth_S
                    H3[old_cols, new_cols] .= zero(T)
                    H3[new_cols, old_cols] .= zero(T)
                else
                    mul!(H3[old_cols, new_cols], J4[:, old_cols]', V_new)
                    H3[new_cols, old_cols] .= H3[old_cols, new_cols]'
                end
            end
        else
            # light recycle cache: use only J5/J6 to update H1/H2/H3
            J5_new_sn = J5[1:sn, new_cols]

            mul!(H1[new_cols, new_cols], J5_new', J5_new)
            mul!(H2[new_cols, new_cols], J5_new_sn', J6[1:sn, new_cols])
            if _orth_S
                H3[new_cols, new_cols] .= zero(T)
                for i in 1:r
                    H3[p*r+i, p*r+i] = one(T)
                end
            else
                mul!(H3[new_cols, new_cols], J6[1:sn, new_cols]', J6[1:sn, new_cols])
            end

            if p > 0
                mul!(H1[old_cols, new_cols], J5[:, old_cols]', J5_new)
                H1[new_cols, old_cols] .= H1[old_cols, new_cols]'

                mul!(H2[old_cols, new_cols], J5[1:sn, old_cols]', J6[1:sn, new_cols])
                mul!(H2[new_cols, old_cols], J5_new_sn', J6[1:sn, old_cols])

                if _orth_S
                    H3[old_cols, new_cols] .= zero(T)
                    H3[new_cols, old_cols] .= zero(T)
                else
                    mul!(H3[old_cols, new_cols], J6[1:sn, old_cols]', J6[1:sn, new_cols])
                    H3[new_cols, old_cols] .= H3[old_cols, new_cols]'
                end
            end
        end
    end
    
    pts[p + 1] = z
    recySp.p = p + 1
end
