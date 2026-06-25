mutable struct LobpcgSvdControl
    m::Int
    n::Int
    r::Int
    maxit::Int
    tol::Float64
    side::Char
    verbose::Int
    normC::Float64
    orth_type::Symbol
    conv_range::UnitRange{Int}
    conv_thresh_type::Int
end

"""
    LobpcgSvdControl(m, n, r; kwargs...)

Validate and store solver dimensions, tolerances, orthogonalization mode, and convergence policy.
"""
function LobpcgSvdControl(
    m::Int,
    n::Int,
    r::Int;
    maxit::Int = 2000,
    tol::Float64 = 1e-3,
    side::Char = 'S',
    verbose::Int = 0,
    normC::Float64 = 1.0,
    orth_type::Symbol = :lapack,
    conv_range::UnitRange{Int} = 1:r,
    conv_thresh_type::Int = 0
)
    1 <= first(conv_range) <= last(conv_range) <= r ||
        throw(ArgumentError("conv_range=$conv_range must satisfy 1 <= first <= last <= r=$r"))
    orth_type in (:lapack, :shiftchol) ||
        throw(ArgumentError("orth_type=$orth_type must be :lapack or :shiftchol"))
    conv_thresh_type in (0, 1) ||
        throw(ArgumentError("conv_thresh_type=$conv_thresh_type must be 0 or 1"))
    return LobpcgSvdControl(
        m,
        n,
        r,
        maxit,
        tol,
        side,
        verbose,
        normC,
        orth_type,
        conv_range,
        conv_thresh_type
    )
end

# Keyword-only constructor that forwards to the validating positional constructor.
LobpcgSvdControl(; m::Int, n::Int, r::Int, kwargs...) = LobpcgSvdControl(m, n, r; kwargs...)

struct LobpcgSvdKeep{T<:Union{Float32,Float64,ComplexF32,ComplexF64}, RT<:Real}
    S::Matrix{T}
    CS::Matrix{T}
    G::Matrix{T}
    R::Matrix{T}
    σ::Vector{RT}
    λ::Vector{RT}
end

"""
    LobpcgSvdKeep(T, m, n, r)

Allocate reusable work arrays for the block LOBPCG singular-value iteration.
"""
function LobpcgSvdKeep(::Type{T}, m::Int, n::Int, r::Int) where T
    RT = real(T)
    S = Matrix{T}(undef, n, 3*r)
    CS = Matrix{T}(undef, m, 3*r)
    G = Matrix{T}(undef, 3*r, 3*r)
    R = Matrix{T}(undef, 3*r, 3*r)
    σ = Vector{RT}(undef, 3*r)
    λ = Vector{RT}(undef, 3*r)
    LobpcgSvdKeep{T, RT}(S, CS, G, R, σ, λ)
end

# Validate that caller-provided work arrays match the solver dimensions.
@inline function _resolve_lobpcg_dims(
    control::LobpcgSvdControl,
    keep::LobpcgSvdKeep
)
    expected_cols = 3 * control.r
    if size(keep.S) != (control.n, expected_cols)
        throw(DimensionMismatch("size(keep.S)=$(size(keep.S)) does not match (n, 3r)=($(control.n), $expected_cols)"))
    end
    if size(keep.CS) != (control.m, expected_cols)
        throw(DimensionMismatch("size(keep.CS)=$(size(keep.CS)) does not match (m, 3r)=($(control.m), $expected_cols)"))
    end
    if size(keep.G) != (expected_cols, expected_cols)
        throw(DimensionMismatch("size(keep.G)=$(size(keep.G)) does not match (3r, 3r)=($expected_cols, $expected_cols)"))
    end
    if size(keep.R) != (expected_cols, expected_cols)
        throw(DimensionMismatch("size(keep.R)=$(size(keep.R)) does not match (3r, 3r)=($expected_cols, $expected_cols)"))
    end
    if length(keep.σ) != expected_cols
        throw(DimensionMismatch("length(keep.σ)=$(length(keep.σ)) does not match 3r=$expected_cols"))
    end
    if length(keep.λ) != expected_cols
        throw(DimensionMismatch("length(keep.λ)=$(length(keep.λ)) does not match 3r=$expected_cols"))
    end

    return control.m, control.n, control.r
end

# Validate and normalize the convergence range inside the leading r columns.
@inline function _resolve_conv_range(
    conv_range::UnitRange{Int},
    r::Int
)
    r > 0 || throw(ArgumentError("r must be positive, got $r"))
    1 <= first(conv_range) <= last(conv_range) <= r ||
        throw(ArgumentError("conv_range=$conv_range must satisfy 1 <= first <= last <= r=$r"))
    return conv_range
end

"""
    svdRaylRitz!(CS, R_view, G_view, sigma, lambda, r, side; orth_type = :lapack)

Perform the SVD Rayleigh-Ritz projection on the current image basis `CS`.
"""
function svdRaylRitz!(
    CS::AbstractMatrix{T},
    R_view::AbstractMatrix{T},
    G_view::AbstractMatrix{T},
    σ::AbstractVector{RT},
    λ::AbstractVector{RT},
    r::Int,
    side::Char;
    orth_type::Symbol = :lapack
) where {T, RT<:Real}
    nS = size(CS, 2)
    
    # Step 1: QR factorize CS in place; store the upper triangle in R.
    # R_view = @view R[1:nS, 1:nS]
    rls_qr!(CS, R_view; type = orth_type)
    
    # Step 2: SVD Rayleigh-Ritz; update the active blocks of G, sigma, and lambda.
    @no_escape begin
        # Copy R because svd! overwrites its input.
        R_work = @alloc(T, nS, nS)
        copyto!(R_work, R_view)
        
        # In-place SVD: rls_svd! returns Vt, then converts it to V when requested.
        rls_svd!(R_work, σ, G_view; return_V = true, ascending = (side != 'L'))

        # Compute sigma and lambda in place.
        if side == 'L'
            # side == 'L': keep the largest singular values and invert them.
            @inbounds for i = 1:r
                σ[i] = one(RT) / σ[i]
                λ[i] = σ[i]^2
            end
        else
            # side != 'L' (that is, 'S'): ascending=true puts the smallest values first.
            @inbounds for i = 1:r
                λ[i] = σ[i]^2
            end
        end
        
        # Compute G_P in place in columns r+1:2r of G.
        if nS > r
            G_rest = @view G_view[:, r+1:nS]
            G_top_rest = @view G_view[1:r, r+1:nS]
            G_P_view = @view G_view[:, r+1:2*r]
            
            # LQ factorization.
            Q_mat = @alloc(T, r, nS - r)
            copyto!(Q_mat, G_top_rest)
            rls_lq!(Q_mat)
            
            # G_P = G_rest * Q'. Use a temporary to avoid aliasing.
            G_P_temp = @alloc(T, nS, r)
            mul!(G_P_temp, G_rest, Q_mat')
            copyto!(G_P_view, G_P_temp)
        end
    end
    
    return nothing
end
    
    

"""
    basisUpdate!(V, CV, S, CS, R_view, G_view)

Rotate the trial basis and its image using the Ritz vectors in `G_view`.
"""
# Update basis vectors and their images with Ritz vectors.
# V_new = V_old * G,  CV_new = CV_old * (R * G)
@inline function basisUpdate!(
    V::AbstractMatrix{T},
    CV::AbstractMatrix{T},
    S::AbstractMatrix{T},
    CS::AbstractMatrix{T},
    R_view::AbstractMatrix{T},
    G_view::AbstractMatrix{T}
) where T
    n = size(V, 1)      
    m = size(CV, 1)
    nV = size(V, 2)    
    nS = size(S, 2)
    @no_escape begin
        temp_V = @alloc(T, n, nV)
        temp_CV = @alloc(T, m, nV)
        temp_RG = @alloc(T, nS, nV)
        # Update basis vectors: V = S * G.
        mul!(temp_V, S, G_view)
        copyto!(V, temp_V)
        
        # Update basis images: CV = CS * (R * G).
        mul!(temp_RG, R_view, G_view)
        mul!(temp_CV, CS, temp_RG)
        copyto!(CV, temp_CV)
    end
end

"""
    residual!(C, V, CV, W, lambda, side)

Compute the normal-equation residuals for the current singular-vector estimates.
"""
function residual!(C::AbstractMatrix{T}, V::AbstractMatrix{T}, CV::AbstractMatrix{T}, W::AbstractMatrix{T}, λ::AbstractVector{S}, side::Char) where {T, S}
    # W = C'C*V - V.*λ (zero-allocation version with SIMD)
    mul!(W, C', CV)
    n, r = size(W)
    if side == 'S'
        # W[:, j] -= V[:, j] * λ[j]
        @inbounds for j = 1:r
            λj = λ[j]
            @simd for i = 1:n
                W[i, j] -= V[i, j] * λj
            end
        end
    elseif side == 'L'
        # W[:, j] = W[:, j] * λ[j] - V[:, j]
        @inbounds for j = 1:r
            λj = λ[j]
            @simd for i = 1:n
                W[i, j] = W[i, j] * λj - V[i, j]
            end
        end
    end
end


"""
    printInfo(sigma, lambda, W, it; io = stdout, log_prefix = "")

Print per-iteration singular values and residual norms for diagnostic output.
"""
function printInfo(
    σ::AbstractVector{T},
    λ::AbstractVector{T},
    W::AbstractMatrix{S},
    it::Int;
    io::IO = stdout,
    log_prefix::AbstractString = ""
) where {T, S}
    r = size(W, 2)
    @views res = [norm(W[:, j]) for j = 1:r]
    resRel = res ./ σ[1:r]
    @printf(io, "%sit = %d -----------------------\n", log_prefix, it)
    @printf(io, "%sσ      = ", log_prefix)
    for i = 1:r
        @printf(io, "%.4e ", σ[i])
    end
    @printf(io, "\n")
    @printf(io, "%sres    = ", log_prefix)
    for i = 1:r
        @printf(io, "%.4e ", res[i])
    end    
    @printf(io, "\n")
    @printf(io, "%sresRel = ", log_prefix)
    for i = 1:r
        @printf(io, "%.4e ", resRel[i])
    end    
    @printf(io, "\n")

end



"""
    conv_thresh(sigma, normC, tol, type)

Compute the residual threshold used by `check_conv_svd`.
"""
# type = 0, normC * σ * tol
# type = 1, normC^2 * tol
function conv_thresh(
    σ::T1,
    normC::T1,
    tol::T1,
    type::Int
) where {T1<:Real}
    thresh = zero(T1)
    if type == 0
        thresh = max(normC * σ * tol, normC^2 * sqrt(2) * eps(T1))
    elseif type == 1
        thresh = max(normC^2 * tol, normC^2 * sqrt(2) * eps(T1))
    else 
        throw(ArgumentError("unsupported conv thresh type: $type"))
    end
    return thresh
end

"""
    check_conv_svd(W, sigma, tol, side; kwargs...)

Return true when all selected residual columns satisfy the chosen convergence criterion.
"""
function check_conv_svd(
    W::AbstractMatrix{T},
    σ::Union{Nothing, AbstractVector{T1}},
    tol::T1,
    side::Char;
    verbose::Int = 0,
    normC::T1 = one(T1),
    conv_range::UnitRange{Int} = 1:size(W, 2),
    conv_thresh_type::Int = 0
) where {T, T1<:Real}

    # TODO for large case
    if side != 'S'
        return norm(@view(W[:, 1])) < tol
    end

    conv_thresh_type in (0, 1) ||
        throw(ArgumentError("conv_thresh_type must be 0 or 1"))

    rng = _resolve_conv_range(conv_range, size(W, 2))

    if conv_thresh_type == 0
        σ === nothing &&
            throw(ArgumentError("σ cannot be nothing when conv_thresh_type == 0"))

        @inbounds for j in rng
            if norm(@view(W[:, j])) >= conv_thresh(σ[j], normC, tol, 0)
                return false
            end
        end
    else
        thresh = conv_thresh(zero(T1), normC, tol, 1)
        @inbounds for j in rng
            if norm(@view(W[:, j])) >= thresh
                return false
            end
        end
    end

    return true
end


"""
    lobpcgSvd(C, X0, K, keep, control; io = stdout, log_prefix = "")

Run block LOBPCG for smallest or largest singular values with optional preconditioning.
"""
function lobpcgSvd(
    C::AbstractMatrix{T},
    X0::Union{Nothing,AbstractMatrix{T}},
    K::AbstractMatrix{T},
    keep::LobpcgSvdKeep{T},
    control::LobpcgSvdControl;
    io::IO = stdout,
    log_prefix::AbstractString = ""
) where T 

    @unpack maxit, tol, side, verbose, normC, orth_type, conv_thresh_type = control
    m, n, r = _resolve_lobpcg_dims(control, keep)
    conv_range = _resolve_conv_range(control.conv_range, r)
    @unpack S, CS, G, R, σ, λ = keep


    @views @inbounds begin
        X = S[:, 1:r]
        P = S[:, r+1:2*r]
        W = S[:, 2*r+1:3*r]
        XP = S[:, 1:2*r]

        CX = CS[:, 1:r]
        CP = CS[:, r+1:2*r]
        CW = CS[:, 2*r+1:3*r]
        CXP = CS[:, 1:2*r]
        
        λ_r = λ[1:r]
        
        # Pre-create views required by basisUpdate!.
        R_r = R[1:r, 1:r]
        G_r_r = G[1:r, 1:r]
        R_2r = R[1:2*r, 1:2*r]
        G_2r_2r = G[1:2*r, 1:2*r]
        R_3r = R[1:3*r, 1:3*r]
        G_3r_2r = G[1:3*r, 1:2*r]
        G_3r_3r = G[1:3*r, 1:3*r]
    end

    isstop = false

    # iteration 1
    if X0 !== nothing
        copyto!(X, X0)
    else 
        rand!(X)
    end
    rls_qr!(X; type = orth_type)
    mul!(CX, C, X)

    svdRaylRitz!(CX, R_r, G_r_r, σ, λ, r, side; orth_type = orth_type)

    basisUpdate!(X, CX, X, CX, R_r, G_r_r)
    
    residual!(C, X, CX, P, λ_r, side)

    isstop = check_conv_svd(
        P,
        σ,
        tol,
        side;
        verbose = verbose,
        normC = normC,
        conv_range = conv_range,
        conv_thresh_type = conv_thresh_type
    )

    ldiv!(K, P)

    rls_orth!(X, P; type = orth_type)

    mul!(CP, C, P)
    it = 0
    
    for i = 1:maxit        
        # Update the basis with Ritz vectors.
        if i == 1
            svdRaylRitz!(CXP, R_2r, G_2r_2r, σ, λ, r, side; orth_type = orth_type)
            basisUpdate!(XP, CXP, XP, CXP, R_2r, G_2r_2r)
        else
            svdRaylRitz!(CS, R_3r, G_3r_3r, σ, λ, r, side; orth_type = orth_type)
            basisUpdate!(XP, CXP, S, CS, R_3r, G_3r_2r)
        end
        
        residual!(C, X, CX, W, λ_r, side)
        if verbose >= 2
            printInfo(σ, λ, W, i; io = io, log_prefix = log_prefix)
        end
        isstop = check_conv_svd(
            W,
            σ,
            tol,
            side;
            verbose = verbose,
            normC = normC,
            conv_range = conv_range,
            conv_thresh_type = conv_thresh_type
        )

        if isstop
            if verbose >= 1
                @printf(io, "%sConverged at iteration %d ", log_prefix, i)
                println(io, "σ[1] = ", side == 'S' ? σ[1] : one(real(T))/σ[1])
            end
            return side == 'S' ? σ[1] :  one(real(T))/σ[1], i
        end
        ldiv!(K, W)

        rls_orth!(XP, W; type = orth_type)
        mul!(CW, C, W)
    end
    return side == 'S' ? σ[1] :  one(real(T))/σ[1], maxit
end


