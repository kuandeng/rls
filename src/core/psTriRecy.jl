"""
    pre_process(C::Matrix)

Build the dense triangular/generalized Schur representation used by psTriRecy.
"""
function pre_process(C::Matrix{T}) where T
    m, n = size(C)
    m >= n || throw(DimensionMismatch("pre_process only supports dense matrices with m >= n, got size $(size(C))"))

    if m == n
        F = schur(Matrix(C))
        M = UpperTriangular(F.T)
        S = Diagonal(fill(one(T), n))
        return M, S
    elseif m >= 2n
        @views C1 = C[1:n, :]
        @views C2 = C[n+1:m, :]
        R2 = UpperTriangular(Matrix(qr(C2).R))
        M = vcat(Matrix(C1), Matrix(R2))
        S = FilledMat(IdentityMat{T}(n), 2n)
        return M, S
    else
        k = m - n
        I_embed = zeros(T, m, n)
        @inbounds for i = 1:n
            I_embed[i, i] = one(T)
        end

        @views I1 = Matrix(I_embed[1:k, :])
        @views I2 = Matrix(I_embed[k+1:m, :])
        @views C1 = Matrix(C[1:k, :])
        @views C2 = Matrix(C[k+1:m, :])

        F = schur(C2, I2)
        Z = Matrix(F.Z)
        M = vcat(C1 * Z, Matrix(F.S))
        S = vcat(I1 * Z, Matrix(F.T))
        return M, S
    end
end

"""
    pre_process(C::SparseMatrixCSC)

Build sparse permuted operators `(M, S)` for shifted sparse solves.
"""
function pre_process(C::SparseArrays.SparseMatrixCSC{T, IT}) where {T, IT}
    m, n = size(C)
    m == n || throw(DimensionMismatch("pre_process only supports square matrices"))
    CH = sparse(adjoint(C))
    N = sparse(LowerTriangular(SparseArrays.spmatmul(CH, C)))
    p = Vector{IT}(amd(N))
    M = C[:, p]
    idx = Vector{IT}(undef, n)
    @inbounds for i = 1:n
        idx[i] = IT(i)
    end
    I_ = sparse(idx, idx, fill(one(T), n), n, n)
    S = I_[:, p]
    return M, S
end

# Count M-family operator applications for the explicit recycled Rayleigh-Ritz path.
@inline function _psTriRecy_explicit_recycle_m_apps(
    r_svdrr::Int,
    r::Int
)
    # Explicit fastSvdRaylRitz_explicit! applies the M-family operator to:
    # 1. the retained recycle block (r_svdrr columns),
    # 2. the recycled singular vectors (r columns),
    # 3. the adjoint-side cache build (r columns).
    return r_svdrr + 2 * r
end

# Count M-family applications needed to insert a refined block into the recycle space.
@inline function _psTriRecy_update_m_apps(
    r::Int,
    implict_recycle::Bool
)
    # updateRecycledSpace!(M, S, ...) always has one M*V_new.
    # The implicit path additionally forms M'*J5 and M'*J6, which are counted
    # in the same M-family total because M and M' are equivalent here.
    return implict_recycle ? (3 * r) : r
end

# Dense helper: R = M - z*S (S diagonal)
function get_R_factor!(
    R::UpperTriangular{T, Matrix{T}},
    M::UpperTriangular{T, Matrix{T}},
    S::Diagonal{T, <:AbstractVector{T}},
    z::T
) where {T}
    size(R) == size(M) || throw(DimensionMismatch("R and M size mismatch"))
    n = size(M, 1)
    @inbounds @simd for j = 1:n
        R[j, j] = M[j, j] - z * S[j, j]
    end
    return nothing
end

# Update dense workspace Mz for a shifted matrix with identity-like S.
function _update_dense_shift_matrix!(
    Mz::Matrix{T},
    M::AbstractMatrix{T},
    S::FilledMat{T, <:IdentityMat{T}},
    z::T
) where {T}
    size(Mz) == size(M) || throw(DimensionMismatch("Mz and M size mismatch"))
    size(M) == size(S) || throw(DimensionMismatch("M and S size mismatch"))
    copyto!(Mz, M)
    n = size(S.A, 1)
    @inbounds @simd for j = 1:n
        Mz[j, j] -= z
    end
    return nothing
end

# Update dense workspace Mz for a general shifted matrix M - zS.
function _update_dense_shift_matrix!(
    Mz::Matrix{T},
    M::AbstractMatrix{T},
    S::AbstractMatrix{T},
    z::T
) where {T}
    size(Mz) == size(M) || throw(DimensionMismatch("Mz and M size mismatch"))
    size(M) == size(S) || throw(DimensionMismatch("M and S size mismatch"))
    copyto!(Mz, M)
    @. Mz -= z * S
    return nothing
end

# Factor the dense shifted matrix M - zS and return its R factor.
function get_R_factor!(
    Mz::Matrix{T},
    M::AbstractMatrix{T},
    S::AbstractMatrix{T},
    z::T
) where {T}
    _update_dense_shift_matrix!(Mz, M, S, z)
    F = qr(Mz)
    return UpperTriangular(Matrix(F.R))
end

# Map each nonzero in A to its location in a superset sparse workspace W.
function _build_nzval_map(
    A::SparseArrays.SparseMatrixCSC{T, IT},
    W::SparseArrays.SparseMatrixCSC{T, IT}
) where {T, IT}
    size(A) == size(W) || throw(DimensionMismatch("A and W size mismatch"))
    n = size(A, 2)
    rowA = rowvals(A)
    rowW = rowvals(W)
    map = Vector{Int}(undef, nnz(A))

    @inbounds for j = 1:n
        pa = A.colptr[j]
        pa_end = A.colptr[j + 1] - 1
        pw = W.colptr[j]
        pw_end = W.colptr[j + 1] - 1

        while pa <= pa_end
            ra = rowA[pa]
            while pw <= pw_end && rowW[pw] < ra
                pw += 1
            end
            if pw > pw_end || rowW[pw] != ra
                throw(ArgumentError("workspace pattern does not contain all nonzeros from source matrix"))
            end
            map[pa] = pw
            pa += 1
        end
    end
    return map
end

# Build a sparse workspace whose pattern is the union of M and S.
function _build_sparse_shift_workspace(
    M::SparseArrays.SparseMatrixCSC{T, IT},
    S::SparseArrays.SparseMatrixCSC{T, IT}
) where {T, IT}
    size(M) == size(S) || throw(DimensionMismatch("M and S size mismatch"))
    m, n = size(M)

    rowM, colM, _ = findnz(M)
    rowS, colS, _ = findnz(S)
    rowU = vcat(rowM, rowS)
    colU = vcat(colM, colS)
    marker = ones(Int, length(rowU))
    pattern = sparse(rowU, colU, marker, m, n)
    rowP, colP, _ = findnz(pattern)

    Mz = sparse(rowP, colP, zeros(T, length(rowP)), m, n)
    mz_map_M = _build_nzval_map(M, Mz)
    mz_map_S = _build_nzval_map(S, Mz)
    return Mz, mz_map_M, mz_map_S
end

# Update sparse workspace Mz in place with values from M - zS.
function _update_sparse_shift_matrix!(
    Mz::SparseArrays.SparseMatrixCSC{T, IT},
    M::SparseArrays.SparseMatrixCSC{T, IT},
    S::SparseArrays.SparseMatrixCSC{T, IT},
    mz_map_M::Vector{Int},
    mz_map_S::Vector{Int},
    z::T
) where {T, IT}
    nz_Mz = nonzeros(Mz)
    nz_M = nonzeros(M)
    nz_S = nonzeros(S)

    fill!(nz_Mz, zero(T))
    @inbounds @simd for k in eachindex(nz_M)
        nz_Mz[mz_map_M[k]] = nz_M[k]
    end
    @inbounds for k in eachindex(nz_S)
        nz_Mz[mz_map_S[k]] -= z * nz_S[k]
    end
    return nothing
end

# Factor the sparse shifted matrix M - zS and return its R factor.
function get_R_factor!(
    Mz::SparseArrays.SparseMatrixCSC{T, IT},
    M::SparseArrays.SparseMatrixCSC{T, IT},
    S::SparseArrays.SparseMatrixCSC{T, IT},
    mz_map_M::Vector{Int},
    mz_map_S::Vector{Int},
    z::T;
    ordering::Int = 0
) where {T, IT}
    _update_sparse_shift_matrix!(Mz, M, S, mz_map_M, mz_map_S, z)
    F = qr(Mz; ordering = ordering)
    return UpperTriangular(F.R)
end

# Check whether the recycled residual is accurate enough to skip refinement.
@inline function _residual_norm_ok(
    W::AbstractMatrix{T},
    σ::Real,
    tol::Real,
    normC::Real
) where {T}
    RT = real(T)
    return norm(@view(W[:, 1])) < normC * max(σ * tol, normC * sqrt(2) * eps(RT))
end

mutable struct PsTriRecyStepState
    recycled::Bool
    p::Int
    refine::Bool
    refine_it::Int
end

# Initialize a per-grid-point state from the current recycle-space size.
PsTriRecyStepState(p::Int = 0) = PsTriRecyStepState(false, p, false, 0)

mutable struct PsTriRecyStats{TZ<:Number}
    p_count::Vector{Int}
    r_recy::Vector{Int}
    pts::Vector{TZ}
    M_vec_apps::Vector{Int}
    M_vec_apps_no_recycle::Vector{Int}
    R_vec_apps::Vector{Int}
    preprocess_time_ns::UInt64
    recycle_time_ns::UInt64
    qr_time_ns::UInt64
    refine_time_ns::UInt64
    update_time_ns::UInt64
    mv_stats_available::Bool
    n_it::Int
    n_pts::Int
end

# Allocate statistics arrays for one psTriRecy grid run.
function PsTriRecyStats(::Type{TZ}, nZ::Int; mv_stats_available::Bool) where {TZ<:Number}
    return PsTriRecyStats{TZ}(
        zeros(Int, nZ),
        zeros(Int, nZ),
        zeros(TZ, nZ),
        zeros(Int, nZ),
        zeros(Int, nZ),
        zeros(Int, nZ),
        zero(UInt64),
        zero(UInt64),
        zero(UInt64),
        zero(UInt64),
        zero(UInt64),
        mv_stats_available,
        0,
        0
    )
end

# Preserve legacy lowercase stats-property aliases.
@inline function Base.getproperty(stats::PsTriRecyStats, name::Symbol)
    if name === :m_vec_apps
        return getfield(stats, :M_vec_apps)
    elseif name === :m_vec_apps_no_recycle
        return getfield(stats, :M_vec_apps_no_recycle)
    elseif name === :r_vec_apps
        return getfield(stats, :R_vec_apps)
    end
    return getfield(stats, name)
end

# Optional timing helpers that compile down to zero when timing is disabled.
@inline _psTriRecy_time_ns_if(enabled::Bool) = enabled ? time_ns() : zero(UInt64)
@inline _psTriRecy_elapsed_ns(enabled::Bool, t_start::UInt64) = enabled ? (time_ns() - t_start) : zero(UInt64)
@inline _psTriRecy_ns_to_s(t::UInt64) = Float64(t) / 1e9

# Convert a step state into a compact log label.
@inline function _ps_tri_recy_step_state(st::PsTriRecyStepState)
    if st.recycled
        return "recycled"
    elseif st.refine
        return "refine($(st.refine_it))"
    else
        return "skip"
    end
end

# Print one psTriRecy grid-step summary line.
function _print_ps_tri_recy_step!(
    io::IO,
    i::Int,
    nZ::Int,
    z,
    st::PsTriRecyStepState;
    r_svdrr::Int = -1,
    σ::Real = NaN
)
    state = _ps_tri_recy_step_state(st)
    rr_info = r_svdrr >= 0 ? @sprintf(" r_svdrr=%3d", r_svdrr) : ""
    σ_info = isfinite(σ) ? @sprintf(" σ=%9.3e", σ) : ""
    @printf(
        io,
        "[%4d/%4d] z=%9.4f%+9.4fim | main: state=%-12s p=%3d%s%s\n",
        i,
        nZ,
        real(z),
        imag(z),
        state,
        st.p,
        rr_info,
        σ_info
    )
    @printf(io, "%s\n", repeat("-", 100))
end

Base.@kwdef struct PsTriRecyControl
    tol::Float64 = 1e-3
    normC::Float64 = 1.0
    shift::Float64 = normC^2 * eps(Float64)*100
    gap_tau::Float64 = 1e-10
    gap_tau_scale::Float64 = 0.01
    recy_tau::Float64 = 1.5
    recycle_fail_dist_thresh::Float64 = NaN

    r::Int = 6
    p::Int = 50
    p_min::Int = 1

    lobpcg_tol_scale::Float64 = 0.1
    lobpcg_side::Char = 'L'

    implict_update::Bool = false
    implict_recycle::Bool = true
    collect_timing::Bool = true
    inner_verbose_override::Union{Nothing, Int} = nothing
end

# Return a temporarily scaled gap threshold while forced-add recovery is active.
@inline function _psTriRecy_gap_tau(
    control::PsTriRecyControl,
    scaled_gap_tau_remaining::Int
)
    return scaled_gap_tau_remaining > 0 ? control.gap_tau * control.gap_tau_scale : control.gap_tau
end

# Consume one remaining scaled-gap step.
@inline function _psTriRecy_consume_scaled_gap_tau(
    scaled_gap_tau_remaining::Int
)
    return scaled_gap_tau_remaining > 0 ? scaled_gap_tau_remaining - 1 : 0
end

# Set the scaled-gap budget after a forced recycle-space insertion.
@inline function _psTriRecy_scaled_gap_tau_budget(
    force_add::Bool,
    rec::Recursion
)
    return force_add ? max(rec.k, 0) : 0
end

mutable struct PsTriRecyKeep{
    T,
    RT,
    MType<:UpperTriangular{T, Matrix{T}},
    SType<:Diagonal{T, Vector{T}},
    RType<:UpperTriangular{T, Matrix{T}},
    RS<:RecycledSpace{T},
    LK<:LobpcgSvdKeep{T, RT},
    XM<:AbstractMatrix{T},
    WM<:AbstractMatrix{T},
    SR<:AbstractVector{RT}
}
    m::Int
    n::Int
    M::MType
    S::SType
    Rz::RType
    recySp::RS
    keep::LK
    X::XM
    W::WM
    σ_recy::SR
end

mutable struct PsTriRecySparseKeep{
    T,
    RT,
    IT<:Integer,
    MType<:SparseArrays.SparseMatrixCSC{T, IT},
    SType<:SparseArrays.SparseMatrixCSC{T, IT},
    MZType<:SparseArrays.SparseMatrixCSC{T, IT},
    RS<:RecycledSpace{T},
    LK<:LobpcgSvdKeep{T, RT},
    XM<:AbstractMatrix{T},
    WM<:AbstractMatrix{T},
    SR<:AbstractVector{RT}
}
    m::Int
    n::Int
    M::MType
    S::SType
    Mz::MZType
    mz_map_M::Vector{Int}
    mz_map_S::Vector{Int}
    recySp::RS
    keep::LK
    X::XM
    W::WM
    σ_recy::SR
end

mutable struct PsTriRecyDenseRectKeep{
    T,
    RT,
    MType<:AbstractMatrix{T},
    SType<:AbstractMatrix{T},
    MZType<:Matrix{T},
    RS<:RecycledSpace{T},
    LK<:LobpcgSvdKeep{T, RT},
    XM<:AbstractMatrix{T},
    WM<:AbstractMatrix{T},
    SR<:AbstractVector{RT}
}
    m::Int
    n::Int
    M::MType
    S::SType
    Mz::MZType
    recySp::RS
    keep::LK
    X::XM
    W::WM
    σ_recy::SR
end

# Return the shifted factor object for dense Schur-triangular keeps.
@inline function _get_shift_factor!(keep_tri::PsTriRecyKeep{T}, z::T) where {T}
    get_R_factor!(keep_tri.Rz, keep_tri.M, keep_tri.S, z)
    return keep_tri.Rz
end

# Return the shifted QR factor for dense rectangular keeps.
@inline function _get_shift_factor!(keep_tri::PsTriRecyDenseRectKeep{T}, z::T) where {T}
    return get_R_factor!(keep_tri.Mz, keep_tri.M, keep_tri.S, z)
end

# Return the shifted sparse QR factor for sparse keeps.
@inline function _get_shift_factor!(keep_tri::PsTriRecySparseKeep{T}, z::T) where {T}
    return get_R_factor!(
        keep_tri.Mz,
        keep_tri.M,
        keep_tri.S,
        keep_tri.mz_map_M,
        keep_tri.mz_map_S,
        z;
        ordering = 0
    )
end

# Dense triangular path reuses an in-place shift and has no QR timing component.
@inline function _get_shift_factor_timed!(
    keep_tri::PsTriRecyKeep{T},
    z::T,
    collect_timing::Bool
) where {T}
    return _get_shift_factor!(keep_tri, z), zero(UInt64)
end

# Dense rectangular path factors through get_R_factor! without separate QR timing.
@inline function _get_shift_factor_timed!(
    keep_tri::PsTriRecyDenseRectKeep{T},
    z::T,
    collect_timing::Bool
) where {T}
    return _get_shift_factor!(keep_tri, z), zero(UInt64)
end

# Sparse path times QR separately because it dominates shifted-factor setup.
@inline function _get_shift_factor_timed!(
    keep_tri::PsTriRecySparseKeep{T},
    z::T,
    collect_timing::Bool
) where {T}
    _update_sparse_shift_matrix!(
        keep_tri.Mz,
        keep_tri.M,
        keep_tri.S,
        keep_tri.mz_map_M,
        keep_tri.mz_map_S,
        z
    )
    qr_t_start = _psTriRecy_time_ns_if(collect_timing)
    F = qr(keep_tri.Mz; ordering = 0)
    qr_time_ns = _psTriRecy_elapsed_ns(collect_timing, qr_t_start)
    return UpperTriangular(F.R), qr_time_ns
end

# Map outer verbosity to inner LOBPCG verbosity unless explicitly overridden.
@inline function _resolve_tri_inner_verbose(
    control::PsTriRecyControl,
    verbose::Int
)
    if control.inner_verbose_override !== nothing
        return control.inner_verbose_override::Int
    end
    return verbose >= 4 ? 2 : (verbose >= 3 ? 1 : 0)
end

# Effective row count of a FilledMat right factor.
@inline function _effective_s_rows(S::FilledMat)
    return size(S.A, 1)
end

# Effective row count for ordinary matrix right factors.
@inline function _effective_s_rows(S::AbstractMatrix)
    return size(S, 1)
end

# Print the grid and control settings for a psTriRecy run.
function _print_psTriRecy_startup!(
    io::IO,
    rec::Recursion,
    control::PsTriRecyControl
)
    n_pts = rec.n_x * rec.n_y
    @printf(
        io,
        "[psTriRecy] rec: x_range=[%.6f, %.6f], y_range=[%.6f, %.6f], n_x=%d, n_y=%d, k=%d, n_pts=%d\n",
        rec.x_st,
        rec.x_ed,
        rec.y_st,
        rec.y_ed,
        rec.n_x,
        rec.n_y,
        rec.k,
        n_pts
    )
    print(io, "[psTriRecy] control:\n")
    for name in fieldnames(typeof(control))
        println(io, "  ", name, " = ", repr(getfield(control, name)))
    end
    return nothing
end

# Initialize dense psTriRecy workspaces and solver controls.
function _init_psTriRecy_keep(
    C::Matrix{T},
    control::PsTriRecyControl,
    inner_verbose::Int
) where {T}
    preprocess_t_start = _psTriRecy_time_ns_if(control.collect_timing)
    M, S = pre_process(C)
    preprocess_time_ns = _psTriRecy_elapsed_ns(control.collect_timing, preprocess_t_start)
    m_eff, n = size(M)
    sn = _effective_s_rows(S)

    recySp = RecycledSpace(
        T,
        m_eff,
        n,
        control.p,
        control.r;
        sn = sn,
        p_min = control.p_min,
        orth_S = true,
        implict_update = control.implict_update,
        implict_recycle = control.implict_recycle
    )

    keep = LobpcgSvdKeep(T, n, n, control.r)
    @views X = keep.S[:, 1:control.r]
    @views W = keep.S[:, control.r+1:2control.r]
    rand!(X)
    σ_recy = zeros(real(T), control.r)

    keep_tri = if M isa UpperTriangular{T, Matrix{T}} && S isa Diagonal{T, Vector{T}}
        Rz = copy(M)
        PsTriRecyKeep(
            m_eff,
            n,
            M,
            S,
            Rz,
            recySp,
            keep,
            X,
            W,
            σ_recy
        )
    else
        Mz = zeros(T, size(M))
        PsTriRecyDenseRectKeep(
            m_eff,
            n,
            M,
            S,
            Mz,
            recySp,
            keep,
            X,
            W,
            σ_recy
        )
    end
    control_main = LobpcgSvdControl(
        m = n,
        n = n,
        r = control.r,
        conv_range = 1:1,
        tol = float(control.lobpcg_tol_scale * control.tol),
        side = control.lobpcg_side,
        verbose = inner_verbose,
        normC = control.normC
    )
    return keep_tri, control_main, preprocess_time_ns
end

# Initialize sparse psTriRecy workspaces and solver controls.
function _init_psTriRecy_keep(
    C::SparseArrays.SparseMatrixCSC{T, IT},
    control::PsTriRecyControl,
    inner_verbose::Int
) where {T, IT}
    m, n = size(C)
    M, S = pre_process(C)
    Mz, mz_map_M, mz_map_S = _build_sparse_shift_workspace(M, S)

    recySp = RecycledSpace(
        T,
        m,
        n,
        control.p,
        control.r;
        p_min = control.p_min,
        orth_S = true,
        implict_update = control.implict_update,
        implict_recycle = control.implict_recycle
    )

    keep = LobpcgSvdKeep(T, n, n, control.r)
    @views X = keep.S[:, 1:control.r]
    @views W = keep.S[:, control.r+1:2control.r]
    rand!(X)
    σ_recy = zeros(real(T), control.r)

    keep_tri = PsTriRecySparseKeep(
        m,
        n,
        M,
        S,
        Mz,
        mz_map_M,
        mz_map_S,
        recySp,
        keep,
        X,
        W,
        σ_recy
    )
    control_main = LobpcgSvdControl(
        m = n,
        n = n,
        r = control.r,
        conv_range = 1:1,
        tol = float(control.lobpcg_tol_scale * control.tol),
        side = control.lobpcg_side,
        verbose = inner_verbose,
        normC = control.normC
    )
    return keep_tri, control_main, zero(UInt64)
end

# Shared implementation for all psTriRecy matrix representations.
function _psTriRecy_impl(
    C::AbstractMatrix{T},
    rec::Recursion;
    control::PsTriRecyControl = PsTriRecyControl(),
    verbose::Int = 1,
    io::IO = stdout
) where {T}
    inner_verbose = _resolve_tri_inner_verbose(control, verbose)
    _print_psTriRecy_startup!(io, rec, control)
    keep_tri, control_main, preprocess_time_ns = _init_psTriRecy_keep(C, control, inner_verbose)
    shift = T(control.shift)
    r = control.r
    collect_timing = control.collect_timing
    recycle_fail_dist_thresh = _resolve_recycle_fail_dist_thresh(control.recycle_fail_dist_thresh, rec)
    scaled_gap_tau_remaining = 0

    # convergence history
    nZ = length(rec.Z)
    σ_min = fill(NaN, nZ)
    stats = PsTriRecyStats(
        eltype(rec.Z),
        nZ;
        mv_stats_available = true
    )
    stats.preprocess_time_ns = preprocess_time_ns
    last_failed_main_z::Union{Nothing, T} = nothing

    for i = 1:nZ
        z = rec.Z[i]
        # gap_i = ((control.normC + abs(z))^2 / control.gap_tau) * eps(Float64)
        stats.p_count[i] = keep_tri.recySp.p
        step_st = PsTriRecyStepState(keep_tri.recySp.p)
        r_svdrr = 0
        m_apps_i = 0
        m_apps_no_recycle_i = 0
        r_apps_i = 0

        if keep_tri.recySp.p > 0
            recycle_t_start = _psTriRecy_time_ns_if(collect_timing)
            gap_tau_i = _psTriRecy_gap_tau(control, scaled_gap_tau_remaining)
            if control.implict_recycle
                r_svdrr = fastSvdRaylRitz_implicit!(
                    keep_tri.recySp,
                    z,
                    keep_tri.X,
                    keep_tri.W,
                    keep_tri.σ_recy;
                    shift = shift,
                    τ = gap_tau_i
                )
            else
                r_svdrr = fastSvdRaylRitz_explicit!(
                    keep_tri.recySp,
                    z,
                    keep_tri.X,
                    keep_tri.W,
                    keep_tri.σ_recy,
                    keep_tri.M,
                    keep_tri.S;
                    shift = shift,
                    τ = gap_tau_i
                )
                m_apps_i += _psTriRecy_explicit_recycle_m_apps(r_svdrr, r)
            end
            scaled_gap_tau_remaining = _psTriRecy_consume_scaled_gap_tau(scaled_gap_tau_remaining)
            stats.recycle_time_ns += _psTriRecy_elapsed_ns(collect_timing, recycle_t_start)
            stats.r_recy[i] = r_svdrr
            step_st.recycled = _residual_norm_ok(keep_tri.W, keep_tri.σ_recy[1], control.tol, control.normC)
            if step_st.recycled
                σ_min[i] = keep_tri.σ_recy[1]
                stats.M_vec_apps[i] = m_apps_i
                stats.M_vec_apps_no_recycle[i] = m_apps_no_recycle_i
                stats.R_vec_apps[i] = r_apps_i
                step_st.p = keep_tri.recySp.p
                verbose >= 1 && _print_ps_tri_recy_step!(
                    io,
                    i,
                    nZ,
                    z,
                    step_st;
                    r_svdrr = r_svdrr,
                    σ = σ_min[i]
                )
                continue
            end
        end

        refine_t_start = _psTriRecy_time_ns_if(collect_timing)
        Rz, qr_time_ns = _get_shift_factor_timed!(keep_tri, z, collect_timing)
        iRzT = InvMat(adjoint(Rz))

        step_st.refine = true
        ~, it = lobpcgSvd(
            iRzT,
            keep_tri.X,
            IdentityMat{T}(keep_tri.n),
            keep_tri.keep,
            control_main;
            io = io,
            log_prefix = "[main-lobpcg] "
        )
        σ_min[i] = keep_tri.keep.σ[1]
        step_st.refine_it = it
        if it > 0
            r_apps_i += 2 * r + it * r + (it - 1) * r
        end
        refine_elapsed_ns = _psTriRecy_elapsed_ns(collect_timing, refine_t_start)
        stats.qr_time_ns += qr_time_ns
        stats.refine_time_ns += refine_elapsed_ns - qr_time_ns

        stats.n_it += it
        stats.n_pts += 1
        stats.pts[stats.n_pts] = rec.Z[i]

        if i < nZ && it > 0
            update_t_start = _psTriRecy_time_ns_if(collect_timing)
            update_m_apps = _psTriRecy_update_m_apps(r, control.implict_recycle)
            force_add_main = last_failed_main_z !== nothing &&
                abs(z - last_failed_main_z) <= recycle_fail_dist_thresh
            updateRecycledSpace!(
                keep_tri.M,
                keep_tri.S,
                z,
                keep_tri.recySp,
                @view(keep_tri.keep.S[:, 1:r]);
                τ = control.recy_tau,
                r_sub = 1,
                force_add = force_add_main
            )
            scaled_gap_tau_remaining = _psTriRecy_scaled_gap_tau_budget(force_add_main, rec)
            # The no-recycle subtotal excludes only the explicit recycle reuse step.
            # Refine-triggered recycle-space updates remain part of the singular solve,
            # so they are kept in both totals.
            m_apps_i += update_m_apps
            m_apps_no_recycle_i += update_m_apps
            stats.update_time_ns += _psTriRecy_elapsed_ns(collect_timing, update_t_start)
        end
        if step_st.refine
            last_failed_main_z = z
        end

        stats.M_vec_apps[i] = m_apps_i
        stats.M_vec_apps_no_recycle[i] = m_apps_no_recycle_i
        stats.R_vec_apps[i] = r_apps_i
        step_st.p = keep_tri.recySp.p
        if verbose >= 1
            _print_ps_tri_recy_step!(io, i, nZ, z, step_st; r_svdrr = r_svdrr, σ = σ_min[i])
        end
    end

    println(io, "Total M vec apps for singular value problem: ", sum(stats.M_vec_apps))
    if !control.implict_recycle
        println(
            io,
            "Total M vec apps for singular value problem without recycle stage: ",
            sum(stats.M_vec_apps_no_recycle)
        )
    end
    if collect_timing
        if stats.preprocess_time_ns > 0
            @printf(io, "Total preprocess time for singular value problem: %.6f seconds\n", _psTriRecy_ns_to_s(stats.preprocess_time_ns))
        end
        @printf(io, "Total recycle time for singular value problem: %.6f seconds\n", _psTriRecy_ns_to_s(stats.recycle_time_ns))
        if stats.qr_time_ns > 0
            @printf(io, "Total QR time for singular value problem: %.6f seconds\n", _psTriRecy_ns_to_s(stats.qr_time_ns))
        end
        @printf(io, "Total refine time for singular value problem: %.6f seconds\n", _psTriRecy_ns_to_s(stats.refine_time_ns))
        @printf(io, "Total update time for singular value problem: %.6f seconds\n", _psTriRecy_ns_to_s(stats.update_time_ns))
    end
    println(io, "Total R vec apps for singular value problem: ", sum(stats.R_vec_apps))
    println(io, "Total it for singular value problem: ", stats.n_it)
    println(io, "Total pt for singular value problem: ", stats.n_pts)
    return σ_min, stats
end

"""
    psTriRecy(C, rec; control = PsTriRecyControl(), verbose = 1, io = stdout)

Evaluate the smallest singular value over a recursion grid using triangular
preprocessing and recycled subspaces. Returns `(sigma_min, stats)`.
"""
function psTriRecy(
    C::Matrix{T},
    rec::Recursion;
    kwargs...
) where {T}
    return _psTriRecy_impl(C, rec; kwargs...)
end

# Sparse overload keeps sparse preprocessing and sparse shifted QR.
function psTriRecy(
    C::SparseArrays.SparseMatrixCSC{T, IT},
    rec::Recursion;
    kwargs...
) where {T, IT}
    return _psTriRecy_impl(C, rec; kwargs...)
end

# Fallback overload materializes unsupported matrix types as dense matrices.
function psTriRecy(
    C::AbstractMatrix{T},
    rec::Recursion;
    kwargs...
) where {T}
    return psTriRecy(Matrix(C), rec; kwargs...)
end


# C = rand(100, 100)
# m, n = size(C)
# C = ComplexF64.(C)

# x_st, x_ed = 0.1, 0.2
# y_st, y_ed = 0.1, 0.2
# n_x, n_y = 20, 20
# k = 20
# rec = recursion_grid(x_st, x_ed, y_st, y_ed, n_x, n_y, k)

# control = PsTriRecyControl(tol = 1e-3, r = 6, p = 80, implict_update = false)
# @profview psTriRecy(C, rec; control = control, verbose = 1)
