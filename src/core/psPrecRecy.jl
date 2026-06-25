
# Choose the leading prediction Ritz values that participate in convergence checks.
@inline function _pred_recycle_conv_range(
    λ::AbstractVector{<:Real},
    r::Int,
    idx_cutoff::Real
)
    upper = min(r, length(λ))
    upper >= 1 || throw(ArgumentError("prediction recycle range requires at least one Ritz value"))
    idx = 1
    @inbounds for j = 2:upper
        if λ[j] < idx_cutoff
            idx = j
        else
            break
        end
    end
    return 1:idx
end


"""
    get_pred_M_S_Cz(C, z, K)

Build preconditioned prediction-stage operators based on the current sparse factor `K`.
"""
function get_pred_M_S_Cz(
    C::SparseMatrixCSC{T, IT},
    z::T,
    K::SparseLLTP{T, IT}
) where {T, IT}
    m, n = size(C)
    m == n || throw(DimensionMismatch("get_pred_M_S_Cz only supports square matrices, got size $(size(C))"))
    invF = K.invF
    S = FilledMat(IdentityMat{T}(n), m)
    S_pred = MulChainMat(S, invF)
    M_pred = MulChainMat(C, invF)
    Cz = ShiftedMat(C, -z)
    Cz_pred = MulChainMat(Cz, invF)
    return M_pred, S_pred, Cz_pred
end

# Count main-stage M-family applications for explicit recycled Rayleigh-Ritz.
@inline function _psPrecRecy_main_explicit_recycle_m_apps(
    r_svdrr::Int,
    r::Int
)
    # Explicit fastSvdRaylRitz_explicit! applies the main operator family to:
    # 1. the retained recycle block (r_svdrr columns),
    # 2. the recycled singular vectors (r columns),
    # 3. the adjoint-side cache build (r columns).
    return r_svdrr + 2 * r
end

# Count main-stage M-family applications used to update the recycle cache.
@inline function _psPrecRecy_main_update_m_apps(
    r::Int,
    implict_recycle::Bool
)
    # updateRecycledSpace!(M, S, ...) always has one M*V_new.
    # The implicit path additionally forms M'*J5 and M'*J6, which are counted
    # in the same M-family total because M and M' are equivalent in this bookkeeping.
    return implict_recycle ? (3 * r) : r
end

# Count prediction-stage M/L-family applications during LOBPCG refinement.
@inline function _psPrecRecy_pred_refine_ml_apps(
    it::Int,
    r::Int
)
    # In the pred stage, Cz_pred = (C - zI) * invF. One lobpcgSvd operator
    # application therefore contributes one C/C' application and one invF/invF'
    # application. Over t iterations, lobpcgSvd applies Cz_pred/Cz_pred' a total
    # of 2 * (t + 1) times per block column.
    return it > 0 ? (2 * (it + 1) * r) : 0
end

# Count prediction-stage M/L-family applications during recycle-space updates.
@inline function _psPrecRecy_pred_update_ml_apps(
    r::Int,
    implict_recycle::Bool
)
    # Pred update uses M_pred = C * invF and S_pred = I * invF.
    #
    # Base update path:
    # - mul_coupled_ms! computes invF * V_new once and reuses it for both M_pred
    #   and S_pred, so it contributes +1 L-family app and +1 M-family app.
    #
    # Additional implicit-cache path:
    # - M' * J5 and M' * J6 contribute +2 M-family apps and +2 L-family apps,
    #   because M' = invF' * C'.
    # - S' * J5 and S' * J6 contribute +2 L-family apps, because S' = invF'.
    #
    # The subsequent E_pred = invF * X_pred is counted separately.
    if implict_recycle
        return 3 * r, 5 * r
    else
        return r, r
    end
end

# Count the final embedding E_pred = invF * X_pred after prediction updates.
@inline function _psPrecRecy_pred_embed_l_apps(r::Int)
    # After updating the recycle space we explicitly form E_pred = invF * X_pred.
    return r
end

# Optional timing helpers that compile down to zero when timing is disabled.
@inline _psPrecRecy_time_ns_if(enabled::Bool) = enabled ? time_ns() : zero(UInt64)
@inline _psPrecRecy_elapsed_ns(enabled::Bool, t_start::UInt64) = enabled ? (time_ns() - t_start) : zero(UInt64)
@inline _psPrecRecy_ns_to_s(t::UInt64) = Float64(t) / 1e9

mutable struct PsecompStepState
    recycled::Bool
    p::Int
    refine::Bool
    refine_it::Int
end

# Initialize a per-grid-point state from the current recycle-space size.
PsecompStepState(p::Int = 0) = PsecompStepState(false, p, false, 0)

# Convert a main or prediction step state into a compact log label.
@inline function _step_state(st::PsecompStepState)
    if st.recycled
        return "recycled"
    elseif st.refine
        return "refine($(st.refine_it))"
    else
        return "skip"
    end
end

# Print one psPrecRecy grid-step summary line.
function _print_psPrecRecy_step!(
    io::IO,
    i::Int,
    nZ::Int,
    z,
    pred::PsecompStepState,
    main::PsecompStepState;
    r_svdrr::Int = -1,
    σ_main::Real = NaN,
    λ_upper::Real = NaN,
    λ_lower::Real = NaN
)
    pred_state = _step_state(pred)
    main_state = _step_state(main)
    main_rr = r_svdrr >= 0 ? @sprintf(" r_svdrr=%3d", r_svdrr) : ""
    main_sigma = isfinite(σ_main) ? @sprintf(" σ=%9.3e", σ_main) : ""
    show_lambda = main.refine && isfinite(λ_upper) && isfinite(λ_lower)
    pred_lambda = show_lambda ?
        @sprintf(" λ_upper=%9.4f λ_lower=%9.4f", λ_upper, λ_lower) :
        ""
    @printf(
        io,
        "[%4d/%4d] z=%9.4f%+9.4fim | main: state=%-12s p=%3d%s%s\n",
        i,
        nZ,
        real(z),
        imag(z),
        main_state,
        main.p,
        main_rr,
        main_sigma
    )
    @printf(
        io,
        "%s| pred: state=%-12s p=%3d%s\n",
        repeat(" ", 35),
        pred_state,
        pred.p,
        pred_lambda
    )
    @printf(io, "%s\n", repeat("-", 100))
    flush(io)
end

# Print diagnostics for prediction preconditioner construction or rebuild.
@inline function _print_pred_precond_info!(
    io::IO,
    tag::AbstractString,
    z,
    λ_n::Real,
    γ::Real
)
    @printf(
        io,
        "[pred-precond:%s] z=%10.6f%+10.6fim λ_n=%9.3e γ=%9.3e\n",
        tag,
        real(z),
        imag(z),
        λ_n,
        γ
    )
    flush(io)
    return nothing
end

# Decide whether prediction bounds are poor enough to rebuild the preconditioner.
@inline function _need_pred_precond_rebuild(
    main_recycled::Bool,
    λ_lower::Real,
    λ_upper::Real;
    λ_lower_min::Real = 0.1,
    λ_upper_max::Real = 10.0
)
    return !main_recycled && (λ_lower < λ_lower_min || λ_upper > λ_upper_max)
end

# Dynamic prediction tolerance heuristic based on displacement from z_dagger.
@inline function _pred_tol_dynamic(
    dz::Real,
    σL::Real,
    tol_base::Real
)
    return (dz * σL + 1)^2 * tol_base
end

Base.@kwdef struct PsPrecRecyControl
    tol::Float64 = 1e-3
    lobpcg_tol_scale::Float64 = 0.1
    main_normC::Float64 = 10.0
    main_tau::Float64 = 1.666
    shift::Float64 = main_normC^2 * eps(Float64)*100
    gap_tau::Float64 = 1e-10
    recycle_fail_dist_thresh::Float64 = NaN

    tol_pred::Float64 = 1e-3
    lobpcg_pred_tol_scale::Float64 = 1.0
    pred_conv_range::Union{Nothing, UnitRange{Int}} = nothing
    pred_tau::Float64 = 1.666
    idx_cutoff_pred::Float64 = 1e-2
    pred_orth_type::Symbol = :lapack
    pred_conv_thresh_type::Int = 0

    λ_lower_min::Float64 = 0.1
    λ_upper_max::Float64 = 30.0

    r_main::Int = 6
    p_main::Int = 60
    p_min_main::Int = 1
    main_implict_update::Bool = false
    main_implict_recycle::Bool = false
    r_pred::Int = 20
    p_pred::Int = 20
    p_min_pred::Int = 1
    pred_implict_update::Bool = true
    pred_implict_recycle::Bool = true

    hsl_tol::Float64 = 1e-3
    hsl_cntl_1::Float64 = 1e-6
    hsl_cntl_2::Float64 = 1e-5
    hsl_icntl_1::Int = 200
    hsl_icntl_2::Int = 200

    γ_tol::Float64 = 1e-3
    λ_n_tol::Float64 = 1e-3
    inner_verbose_override::Union{Nothing, Int} = nothing
    collect_timing::Bool = true
end

# Print the grid and control settings for a psPrecRecy run.
function _print_psPrecRecy_startup!(
    io::IO,
    rec::Recursion,
    control::PsPrecRecyControl
)
    n_pts = rec.n_x * rec.n_y
    @printf(
        io,
        "[psPrecRecy] rec: x_range=[%.6f, %.6f], y_range=[%.6f, %.6f], n_x=%d, n_y=%d, k=%d, n_pts=%d\n",
        rec.x_st,
        rec.x_ed,
        rec.y_st,
        rec.y_ed,
        rec.n_x,
        rec.n_y,
        rec.k,
        n_pts
    )
    print(io, "[psPrecRecy] control:\n")
    for name in fieldnames(typeof(control))
        println(io, "  ", name, " = ", repr(getfield(control, name)))
    end
    flush(io)
    return nothing
end

mutable struct PsPrecRecyKeep{
    T,
    RT,
    ITP,
    ITK,
    PVec<:AbstractVector{ITP},
    KType<:SparseLLTP{T, ITK},
    KAdType<:SparseLLTP_AD{T, ITK},
    KM<:LobpcgSvdKeep{T, RT},
    KP<:LobpcgSvdKeep{T, RT},
    EP<:AbstractMatrix{T},
    GP<:AbstractMatrix{T},
    FP<:Factorization{T}
}
    p::PVec
    K::KType
    K_ad::KAdType
    E_pred::EP
    GramE_pred::GP
    F_GramE_pred::FP

    recySp_main::RecycledSpace{T}
    keep_main::KM

    recySp_pred::RecycledSpace{T}
    keep_pred::KP
end

mutable struct PsPrecRecyStats{TZ<:Number}
    p_count::Vector{Int}
    pred_p_count::Vector{Int}
    r_recy::Vector{Int}
    pts::Vector{TZ}
    pred_pts::Vector{TZ}
    precond_rebuild_pts::Vector{TZ}
    main_M_vec_apps::Vector{Int}
    main_M_vec_apps_no_recycle::Vector{Int}
    main_L_vec_apps::Vector{Int}
    pred_M_vec_apps::Vector{Int}
    pred_L_vec_apps::Vector{Int}
    main_recycle_time_ns::UInt64
    main_refine_time_ns::UInt64
    main_update_time_ns::UInt64
    pred_recycle_time_ns::UInt64
    pred_refine_time_ns::UInt64
    pred_update_time_ns::UInt64
    precond_time_ns::UInt64
    n_it::Int
    n_pred_it::Int
    n_pts::Int
    n_pred_pts::Int
    n_precond_rebuild_pts::Int
end

# Allocate statistics arrays for one psPrecRecy grid run.
function PsPrecRecyStats(::Type{TZ}, nZ::Int) where {TZ<:Number}
    return PsPrecRecyStats{TZ}(
        zeros(Int, nZ), zeros(Int, nZ), zeros(Int, nZ),
        zeros(TZ, nZ),  zeros(TZ, nZ),  zeros(TZ, nZ),
        zeros(Int, nZ), zeros(Int, nZ), zeros(Int, nZ), zeros(Int, nZ), zeros(Int, nZ),
        zero(UInt64), zero(UInt64), zero(UInt64),
        zero(UInt64), zero(UInt64), zero(UInt64), zero(UInt64),
        0, 0, 0, 0, 0
    )
end

# Record one point where the prediction preconditioner is built or rebuilt.
@inline function _record_pred_precond_point!(
    stats::PsPrecRecyStats{TZ},
    z::TZ
) where {TZ<:Number}
    # Track all points where the level-1 preconditioner is built, including
    # the initial z_dagger, while avoiding duplicate inserts for the same point.
    if stats.n_precond_rebuild_pts > 0 &&
        stats.precond_rebuild_pts[stats.n_precond_rebuild_pts] == z
        return nothing
    end

    stats.n_precond_rebuild_pts += 1
    stats.precond_rebuild_pts[stats.n_precond_rebuild_pts] = z
    return nothing
end

# Control-aware overload for prediction preconditioner rebuild decisions.
@inline function _need_pred_precond_rebuild(
    main_recycled::Bool,
    λ_lower::Real,
    λ_upper::Real,
    control::PsPrecRecyControl
)
    return _need_pred_precond_rebuild(
        main_recycled,
        λ_lower, λ_upper;
        λ_lower_min = control.λ_lower_min, λ_upper_max = control.λ_upper_max
    )
end

# Estimate the inverse factor norm used in prediction-stage norm bounds.
@inline function _compute_γ(
    K::SparseLLTP{T, IT},
    n::Int,
    γ_tol::Real
) where {T, IT<:Integer}
    IL = InvMat(K.L)
    γ, ~ = lobpcgSvd(
        IL,
        nothing,
        IdentityMat{T}(n),
        LobpcgSvdKeep(T, n, n, 1),
        LobpcgSvdControl(m = n, n = n, r = 1, tol = γ_tol, side = 'L', verbose = 0, normC = 1.0, conv_range = 1:1)
    )
    return γ
end

# Estimate the largest eigenvalue of the preconditioned shifted normal operator.
@inline function _compute_λ_n(
    C::SparseMatrixCSC{T, ITC},
    z::T,
    K::SparseLLTP{T, IT},
    λ_n_tol::Real
) where {T, IT<:Integer, ITC<:Integer}
    m, n = size(C)
    Cz = ShiftedMat(C, -z)
    Cz_pred = MulChainMat(Cz, K.invF)
    σ_n, ~ =  lobpcgSvd(
        Cz_pred,
        nothing,
        IdentityMat{T}(n),
        LobpcgSvdKeep(T, m, n, 1),
        LobpcgSvdControl(m = m, n = n, r = 1, tol = λ_n_tol, side = 'L', verbose = 0, normC = 1.0, conv_range = 1:1)
    )
    return σ_n^2
end

# Map outer verbosity to inner LOBPCG verbosity unless explicitly overridden.
@inline function _resolve_inner_verbose(
    control::PsPrecRecyControl,
    verbose::Int
)
    if control.inner_verbose_override !== nothing
        return control.inner_verbose_override::Int
    end
    return verbose >= 4 ? 2 : (verbose >= 3 ? 1 : 0)
end

# Build or rebuild the first-level sparse preconditioner and associated bounds.
@inline function _get_pred_precond!(
    C::SparseMatrixCSC{T, ITC},
    z::T,
    keep::PsPrecRecyKeep,
    control::PsPrecRecyControl
) where {T, ITC<:Integer}
    n = size(C, 2)

    # compute preconditioner K
    keep.K = get_hsl_LLTP(
        C,
        z,
        control.hsl_tol,
        control.hsl_cntl_1,
        control.hsl_cntl_2,
        control.hsl_icntl_1,
        control.hsl_icntl_2,
        keep.p
    )

    # compute λ_n
    λ_n = _compute_λ_n(C, z, keep.K, control.λ_n_tol)

    # compute γ
    γ = _compute_γ(keep.K, n, control.γ_tol)
    return λ_n, γ
end

# Build the adaptive low-rank correction for the two-level preconditioner.
@inline function _get_pred_precond_ad!(
    C::SparseMatrixCSC{T, ITC},
    CzE::AbstractMatrix{T},
    keep::PsPrecRecyKeep
) where {T, ITC<:Integer}

    @assert size(CzE) == (size(C, 1), size(keep.E_pred, 2))

    mul!(keep.GramE_pred, CzE', CzE)

    copyto!(keep.F_GramE_pred.factors, keep.GramE_pred)
    keep.F_GramE_pred = rls_cholesky!(keep.F_GramE_pred.factors)

    keep.K_ad = SparseLLTP_AD(
        keep.K,
        keep.E_pred,
        keep.GramE_pred,
        keep.F_GramE_pred
    )
    return nothing
end


# Initialize main-stage recycled space, LOBPCG keep, and solver control.
function _init_main_stage(
    ::Type{T},
    m::Int,
    n::Int,
    control::PsPrecRecyControl,
    inner_verbose::Int
) where {T}
    p_main = control.p_main
    r_main = control.r_main
    recySp_main = RecycledSpace(
        T,
        m, n,
        p_main, r_main;
        p_min = control.p_min_main,
        orth_S = true,
        implict_update = control.main_implict_update,
        implict_recycle = control.main_implict_recycle
    )
    keep_main = LobpcgSvdKeep(T, n, n, r_main)
    @views X_main = keep_main.S[:, 1:r_main]
    rand!(X_main)
    @views W_main = keep_main.S[:, r_main+1:2r_main]
    @views σ_recy_main = keep_main.σ[1:r_main]
    control_main = LobpcgSvdControl(
        m = n,
        n = n,
        r = r_main,
        conv_range = 1:1,
        tol = float(control.lobpcg_tol_scale * control.tol),
        side = 'S',
        verbose = inner_verbose,
        normC = control.main_normC
    )
    return recySp_main, keep_main, control_main, X_main, W_main, σ_recy_main
end

# Initialize prediction-stage recycled space, LOBPCG keep, and solver control.
function _init_pred_stage(
    ::Type{T},
    m::Int,
    n::Int,
    control::PsPrecRecyControl,
    inner_verbose::Int
) where {T}
    p_pred = control.p_pred
    r_pred = control.r_pred
    recySp_pred = RecycledSpace(
        T,
        m, n,
        p_pred, r_pred;
        p_min = control.p_min_pred,
        orth_S = false,
        implict_update = control.pred_implict_update,
        implict_recycle = control.pred_implict_recycle
    )
    keep_pred = LobpcgSvdKeep(T, n, n, r_pred)
    @views X_pred = keep_pred.S[:, 1:r_pred]
    rand!(X_pred)
    E_pred = zeros(T, n, r_pred)
    @views W_pred = keep_pred.S[:, r_pred+1:2r_pred]
    @views CX_pred = keep_pred.CS[:, 1:r_pred]
    @views λ_recy_pred = keep_pred.λ[1:r_pred]
    GramE_pred = Matrix{T}(I, r_pred, r_pred)
    F_GramE_pred = rls_cholesky!(copy(GramE_pred))
    control_pred = LobpcgSvdControl(
        m = n,
        n = n,
        r = r_pred,
        tol = float(control.tol_pred * control.lobpcg_pred_tol_scale),
        side = 'S',
        verbose = inner_verbose,
        orth_type = control.pred_orth_type,
        conv_range = control.pred_conv_range === nothing ? (1:r_pred) : control.pred_conv_range,
        conv_thresh_type = control.pred_conv_thresh_type
    )
    return recySp_pred, keep_pred, control_pred, X_pred, E_pred, W_pred, CX_pred, λ_recy_pred, GramE_pred, F_GramE_pred
end

# Initialize all psPrecRecy state shared across the grid traversal.
function _init_psPrecRecy_keep(
    C::SparseMatrixCSC{T, ITC},
    rec::Recursion,
    p::AbstractVector{ITP},
    control::PsPrecRecyControl,
    inner_verbose::Int
) where {T, ITC<:Integer, ITP<:Integer}
    m, n = size(C)

    recySp_main, keep_main, control_main, X_main, W_main, σ_recy_main = 
    _init_main_stage(
        T,
        m,
        n,
        control,
        inner_verbose
    )

    recySp_pred, keep_pred, control_pred, X_pred, E_pred, W_pred, CX_pred, λ_recy_pred, GramE_pred, F_GramE_pred =
    _init_pred_stage(
        T,
        m,
        n,
        control,
        inner_verbose
    )

    keep = PsPrecRecyKeep(
        p,
        SparseLLTP{T, ITC}(),
        SparseLLTP_AD{T, ITC}(),
        E_pred,
        GramE_pred,
        F_GramE_pred,
        recySp_main,
        keep_main,
        recySp_pred,
        keep_pred
    )
    return keep, keep_main, keep_pred, control_main, control_pred, X_main, W_main, σ_recy_main, recySp_main, X_pred, E_pred, W_pred, CX_pred, λ_recy_pred, recySp_pred
end


"""
    psPrecRecy(C, rec; control = PsPrecRecyControl(), verbose = 1, io = stdout)

Evaluate the smallest singular value over a recursion grid using a sparse
preconditioner plus recycled main and prediction spaces. Returns `(sigma_min, stats)`.
"""
function psPrecRecy( 
    C::SparseMatrixCSC{T, IT},
    rec::Recursion;
    control::PsPrecRecyControl = PsPrecRecyControl(),
    verbose::Int = 1,
    io::IO = stdout
) where {T, IT<:Integer}

    p = qr(C).cpiv
    m, n = size(C)
    inner_verbose = _resolve_inner_verbose(control, verbose)
    _print_psPrecRecy_startup!(io, rec, control)

    # Initialize working state.
    keep, keep_main, keep_pred, 
    control_main, control_pred, 
    X_main, W_main, σ_recy_main, recySp_main,
    X_pred, E_pred, W_pred, CX_pred, λ_recy_pred, recySp_pred =
        _init_psPrecRecy_keep(C, rec, p, control, inner_verbose)

    r_pred = control.r_pred
    r_main = control.r_main
    collect_timing = control.collect_timing
    recycle_fail_dist_thresh = _resolve_recycle_fail_dist_thresh(control.recycle_fail_dist_thresh, rec)

    # init M S
    M = C
    S = FilledMat(IdentityMat{T}(n), m)

    nZ = length(rec.Z)
    σ_min = fill(NaN, nZ)
    stats = PsPrecRecyStats(eltype(rec.Z), nZ)


    # construct preconditioner
    z_dagger = rec.Z[1]
    precond_t_start = _psPrecRecy_time_ns_if(collect_timing)
    λ_dagger_n, γ = _get_pred_precond!(C, z_dagger, keep, control)
    stats.precond_time_ns += _psPrecRecy_elapsed_ns(collect_timing, precond_t_start)
    _record_pred_precond_point!(stats, z_dagger)
    verbose >= 1 && _print_pred_precond_info!(io, "init", z_dagger, λ_dagger_n, γ)

    λ_upper = 0.0
    λ_lower = 0.0
    last_failed_main_z::Union{Nothing, T} = nothing
    last_failed_pred_z::Union{Nothing, T} = nothing

    
    # main loop
    for i = 1:nZ
        z = rec.Z[i]
        Cz = ShiftedMat(C, -z)
        stats.p_count[i] = recySp_main.p
        pred_st = PsecompStepState(recySp_pred.p)
        stats.pred_p_count[i] = pred_st.p
        main_st = PsecompStepState(recySp_main.p)
        r_svdrr = 0
        main_M_apps_i = 0
        main_M_apps_no_recycle_i = 0
        main_L_apps_i = 0
        pred_M_apps_i = 0
        pred_L_apps_i = 0

        # RecySingularSpace: reuse the main recycled singular space.
        if recySp_main.p > 0
            recycle_t_start = _psPrecRecy_time_ns_if(collect_timing)
            # recycle the singular subspace
            if control.main_implict_recycle
                r_svdrr = fastSvdRaylRitz_implicit!(
                    recySp_main,
                    z,
                    X_main, W_main, σ_recy_main;
                    shift = T(control.shift), τ = control.gap_tau
                )
            else
                r_svdrr = fastSvdRaylRitz_explicit!(
                    recySp_main,
                    z,
                    X_main,
                    W_main,
                    σ_recy_main,
                    M,
                    S;
                    shift = T(control.shift),
                    τ = control.gap_tau
                )
                main_M_apps_i += _psPrecRecy_main_explicit_recycle_m_apps(r_svdrr, r_main)
            end

            # check the converge
            main_st.recycled = check_conv_svd(
                W_main, σ_recy_main,
                control.tol,
                'S';
                normC = control.main_normC,  # TODO normCz 
                conv_range = 1:1
            )
            stats.main_recycle_time_ns += _psPrecRecy_elapsed_ns(collect_timing, recycle_t_start)
        end

        stats.r_recy[i] = r_svdrr
        if main_st.recycled
            main_st.p = recySp_main.p
            σ_min[i] = σ_recy_main[1]
            stats.main_M_vec_apps[i] = main_M_apps_i
            stats.main_M_vec_apps_no_recycle[i] = main_M_apps_no_recycle_i
            stats.main_L_vec_apps[i] = main_L_apps_i
            stats.pred_M_vec_apps[i] = pred_M_apps_i
            stats.pred_L_vec_apps[i] = pred_L_apps_i
            verbose >= 1 && _print_psPrecRecy_step!(
                io,
                i, nZ, z,
                pred_st, main_st;
                r_svdrr = r_svdrr,
                σ_main = σ_min[i]
            )
            continue
        end

        # RecyProjSpace: fastRaylRitz! is the projection-space recycle step.
        dz = abs(z - z_dagger)
        pred_normC = sqrt(λ_dagger_n) + γ * dz
        M_pred, S_pred, Cz_pred = get_pred_M_S_Cz(C, z, keep.K)
        idx_pred = r_pred
        conv_range_pred = 1:r_pred

        if recySp_pred.p > 0
            recycle_t_start = _psPrecRecy_time_ns_if(collect_timing)
            # recycle the projction subspace
            fastRaylRitz!(
                recySp_pred,
                z,
                X_pred, E_pred, W_pred, λ_recy_pred
            )

            # check the conv
            idx_pred = something(findlast(<(control.idx_cutoff_pred), @view λ_recy_pred[1:r_pred]), r_pred)
            conv_range_pred = 1:idx_pred

            pred_st.recycled = check_conv_svd(
                W_pred, nothing,
                control.tol_pred,
                'S';
                normC = pred_normC, 
                conv_range = conv_range_pred,
                conv_thresh_type = 1
            )
            stats.pred_recycle_time_ns += _psPrecRecy_elapsed_ns(collect_timing, recycle_t_start)
        end

        if pred_st.recycled
            # Recycled E_pred already stores invF * basis vectors from the previous
            # update, so the current step only applies Cz once and adds one M-family app.
            mul!(CX_pred, Cz, E_pred)
            pred_M_apps_i += r_pred
            λ_lower = λ_recy_pred[r_pred]
            # update_sparse_lldiv_ad_basis!(keep.K_ad, E_pred, CX_pred)
        else
            pred_st.refine = true
            control_pred.normC = pred_normC
            control_pred.tol = float(control.tol_pred * control.lobpcg_pred_tol_scale)
            control_pred.conv_range = conv_range_pred
            refine_t_start = _psPrecRecy_time_ns_if(collect_timing)
            ~, pred_st.refine_it = lobpcgSvd(
                Cz_pred,
                X_pred,
                IdentityMat{T}(n),
                keep_pred,
                control_pred;
                io = io,
                log_prefix = "[pred-lobpcg] "
            )
            stats.pred_refine_time_ns += _psPrecRecy_elapsed_ns(collect_timing, refine_t_start)
            stats.n_pred_it += pred_st.refine_it
            pred_operator_apps = _psPrecRecy_pred_refine_ml_apps(pred_st.refine_it, r_pred)
            if pred_operator_apps > 0
                pred_M_apps_i += pred_operator_apps
                pred_L_apps_i += pred_operator_apps
            end

            update_t_start = _psPrecRecy_time_ns_if(collect_timing)
            force_add_pred = last_failed_pred_z !== nothing &&
                abs(z - last_failed_pred_z) <= recycle_fail_dist_thresh
            updateRecycledSpace!(
                M_pred,
                S_pred,
                z,
                recySp_pred,
                X_pred;
                τ = control.pred_tau,
                r_sub = idx_pred,
                force_add = force_add_pred
            )
            pred_update_M_apps, pred_update_L_apps =
                _psPrecRecy_pred_update_ml_apps(r_pred, control.pred_implict_recycle)
            pred_M_apps_i += pred_update_M_apps
            pred_L_apps_i += pred_update_L_apps

            mul!(E_pred, keep.K.invF, X_pred)
            # Count the final E_pred = invF * X_pred separately so the helper above
            # matches updateRecycledSpace! itself, not the post-update embedding.
            pred_L_apps_i += _psPrecRecy_pred_embed_l_apps(r_pred)
            stats.pred_update_time_ns += _psPrecRecy_elapsed_ns(collect_timing, update_t_start)
            λ_lower = keep_pred.λ[r_pred]
            # update_sparse_lldiv_ad_basis!(keep.K_ad, E_pred, CX_pred)
        end

        λ_upper = pred_normC^2
        if _need_pred_precond_rebuild(main_st.recycled, λ_lower, λ_upper, control)
            _record_pred_precond_point!(stats, z)

            # recompute the preconditioner
            precond_t_start = _psPrecRecy_time_ns_if(collect_timing)
            λ_dagger_n, γ = _get_pred_precond!(C, z, keep, control)
            stats.precond_time_ns += _psPrecRecy_elapsed_ns(collect_timing, precond_t_start)
            verbose >= 1 && _print_pred_precond_info!(io, "rebuild", z, λ_dagger_n, γ)

            # reset params
            M_pred, S_pred, Cz_pred = get_pred_M_S_Cz(C, z, keep.K)
            recySp_pred.p = 0
            rand!(X_pred)
            λ_lower_prev = λ_lower
            λ_upper_prev = λ_upper
            λ_upper = λ_dagger_n

            # Rebuild restarts the pred stage with a fresh refine solve,
            # so the step state must reflect refine rather than recycle.
            pred_st.recycled = false
            pred_st.refine = true

            # compute the projection subspace
            control_pred.normC = sqrt(λ_dagger_n)
            control_pred.tol = float(control.tol_pred * control.lobpcg_pred_tol_scale)
            conv_range_pred = 1:r_pred
            control_pred.conv_range = conv_range_pred
            refine_t_start = _psPrecRecy_time_ns_if(collect_timing)
            ~, pred_st.refine_it = lobpcgSvd(
                Cz_pred,
                X_pred,
                IdentityMat{T}(n),
                keep_pred,
                control_pred;
                io = io,
                log_prefix = "[pred-lobpcg] "
            )
            stats.pred_refine_time_ns += _psPrecRecy_elapsed_ns(collect_timing, refine_t_start)
            stats.n_pred_it += pred_st.refine_it
            pred_operator_apps = _psPrecRecy_pred_refine_ml_apps(pred_st.refine_it, r_pred)
            if pred_operator_apps > 0
                pred_M_apps_i += pred_operator_apps
                pred_L_apps_i += pred_operator_apps
            end

            idx_pred = something(findlast(<(control.idx_cutoff_pred), @view λ_recy_pred[1:r_pred]), r_pred)
            update_t_start = _psPrecRecy_time_ns_if(collect_timing)
            force_add_pred = last_failed_pred_z !== nothing &&
                abs(z - last_failed_pred_z) <= recycle_fail_dist_thresh
            updateRecycledSpace!(
                M_pred,
                S_pred,
                z,
                recySp_pred,
                X_pred;
                τ = control.pred_tau,
                r_sub = idx_pred,
                force_add = force_add_pred
            )
            pred_update_M_apps, pred_update_L_apps =
                _psPrecRecy_pred_update_ml_apps(r_pred, control.pred_implict_recycle)
            pred_M_apps_i += pred_update_M_apps
            pred_L_apps_i += pred_update_L_apps
            mul!(E_pred, keep.K.invF, X_pred)
            # Count the final E_pred = invF * X_pred separately so the helper above
            # matches updateRecycledSpace! itself, not the post-update embedding.
            pred_L_apps_i += _psPrecRecy_pred_embed_l_apps(r_pred)
            stats.pred_update_time_ns += _psPrecRecy_elapsed_ns(collect_timing, update_t_start)
            λ_lower = keep_pred.λ[r_pred]

            if verbose >= 1
                @printf(io, "γ(rebuild): %.6e\n", γ)
                @printf(
                    io,
                    "[pred-rebuild] z_curr=%10.6f%+10.6fim -> %10.6f%+10.6fim | prev λ_lower=%9.3e prev λ_upper=%9.3e | new λ_lower=%9.3e new λ_upper=%9.3e\n",
                    real(z_dagger), imag(z_dagger),
                    real(z), imag(z),
                    λ_lower_prev, λ_upper_prev,
                    λ_lower, λ_upper
                )
                flush(io)
            end
            z_dagger = z
        end

        pred_st.p = recySp_pred.p
        stats.pred_p_count[i] = pred_st.p
        if pred_st.refine
            stats.n_pred_pts += 1
            stats.pred_pts[stats.n_pred_pts] = z
            last_failed_pred_z = z
        end

        # construct two-level precondtioner
        precond_t_start = _psPrecRecy_time_ns_if(collect_timing)
        _get_pred_precond_ad!(C, CX_pred, keep)
        stats.precond_time_ns += _psPrecRecy_elapsed_ns(collect_timing, precond_t_start)

        # Main solve with two-level preconditioner.
        main_st.refine = true
        refine_t_start = _psPrecRecy_time_ns_if(collect_timing)
        σ_main_i, main_st.refine_it = lobpcgSvd(
            Cz,
            X_main,
            keep.K_ad,
            keep_main,
            control_main;
            io = io,
            log_prefix = "[main-lobpcg] "
        )
        stats.main_refine_time_ns += _psPrecRecy_elapsed_ns(collect_timing, refine_t_start)
        σ_min[i] = σ_main_i
        stats.n_it += main_st.refine_it
        if main_st.refine_it > 0
            main_operator_apps = 2 * (main_st.refine_it + 1) * r_main
            main_M_apps_i += main_operator_apps
            main_M_apps_no_recycle_i += main_operator_apps
            main_L_apps_i += 2 * main_st.refine_it * r_main
        end
        stats.n_pts += 1
        stats.pts[stats.n_pts] = rec.Z[i]

        if main_st.refine_it > 0
            update_m_apps = _psPrecRecy_main_update_m_apps(r_main, control.main_implict_recycle)
            update_t_start = _psPrecRecy_time_ns_if(collect_timing)
            force_add_main = last_failed_main_z !== nothing &&
                abs(z - last_failed_main_z) <= recycle_fail_dist_thresh
            updateRecycledSpace!(
                M,
                S,
                z,
                recySp_main,
                X_main;
                τ = control.main_tau,
                force_add = force_add_main
            )
            # The no-recycle subtotal excludes only the explicit recycle reuse step.
            # Refine-triggered recycle-space updates remain part of the singular solve,
            # so they are kept in both totals.
            stats.main_update_time_ns += _psPrecRecy_elapsed_ns(collect_timing, update_t_start)
            main_M_apps_i += update_m_apps
            main_M_apps_no_recycle_i += update_m_apps
        end
        if main_st.refine
            last_failed_main_z = z
        end
        main_st.p = recySp_main.p
        stats.main_M_vec_apps[i] = main_M_apps_i
        stats.main_M_vec_apps_no_recycle[i] = main_M_apps_no_recycle_i
        stats.main_L_vec_apps[i] = main_L_apps_i
        stats.pred_M_vec_apps[i] = pred_M_apps_i
        stats.pred_L_vec_apps[i] = pred_L_apps_i

        if verbose >= 1
            _print_psPrecRecy_step!(
                io,
                i,
                nZ,
                z,
                pred_st,
                main_st;
                r_svdrr = r_svdrr,
                σ_main = σ_min[i],
                λ_upper = λ_upper,
                λ_lower = λ_lower
            )
        end
    end

    println(io, "Total M vec apps for pred singular value problem: ", sum(stats.pred_M_vec_apps))
    println(io, "Total L vec apps for pred singular value problem: ", sum(stats.pred_L_vec_apps))
    println(io, "Total it for pred singular value problem: ", stats.n_pred_it)
    println(io, "Total pt for pred singular value problem: ", stats.n_pred_pts)
    println(io, "Total preconditioner points: ", stats.n_precond_rebuild_pts)
    if collect_timing
        @printf(io, "Total preconditioner time for singular value problem: %.6f seconds\n", _psPrecRecy_ns_to_s(stats.precond_time_ns))
        pred_total_time_ns =
            stats.pred_recycle_time_ns +
            stats.pred_refine_time_ns +
            stats.pred_update_time_ns
        @printf(io, "Total recycle time for pred stage: %.6f seconds\n", _psPrecRecy_ns_to_s(stats.pred_recycle_time_ns))
        @printf(io, "Total refine time for pred stage: %.6f seconds\n", _psPrecRecy_ns_to_s(stats.pred_refine_time_ns))
        @printf(io, "Total update time for pred stage: %.6f seconds\n", _psPrecRecy_ns_to_s(stats.pred_update_time_ns))
        @printf(io, "Total pred-stage time for singular value problem: %.6f seconds\n", _psPrecRecy_ns_to_s(pred_total_time_ns))
    end
    println(io, "Total M vec apps for singular value problem: ", sum(stats.main_M_vec_apps))
    if !control.main_implict_recycle
        println(
            io,
            "Total M vec apps for singular value problem without recycle stage: ",
            sum(stats.main_M_vec_apps_no_recycle)
        )
    end
    println(io, "Total L vec apps for singular value problem: ", sum(stats.main_L_vec_apps))
    println(io, "Total it for singular value problem: ", stats.n_it)
    println(io, "Total pt for singular value problem: ", stats.n_pts)
    if collect_timing
        main_total_time_ns =
            stats.main_recycle_time_ns +
            stats.main_refine_time_ns +
            stats.main_update_time_ns
        @printf(io, "Total recycle time for main stage: %.6f seconds\n", _psPrecRecy_ns_to_s(stats.main_recycle_time_ns))
        @printf(io, "Total refine time for main stage: %.6f seconds\n", _psPrecRecy_ns_to_s(stats.main_refine_time_ns))
        @printf(io, "Total update time for main stage: %.6f seconds\n", _psPrecRecy_ns_to_s(stats.main_update_time_ns))
        @printf(io, "Total main-stage time for singular value problem: %.6f seconds\n", _psPrecRecy_ns_to_s(main_total_time_ns))
    end
    flush(io)
    return σ_min, stats
end

# Dense and generic matrices are materialized as sparse before running psPrecRecy.
function psPrecRecy(
    C::AbstractMatrix{T},
    rec::Recursion;
    kwargs...
) where {T}
    return psPrecRecy(sparse(C), rec; kwargs...)
end
