"""
    affine_conbine!(Az, z, A1, A2, A3, A4; shift = zero(z))

Form `A1 - z*A2 - conj(z)*A3 + (abs2(z)+shift)*A4` in place.
"""
function affine_conbine!(
    Az::AbstractMatrix{T},
    z::T,
    A1::AbstractMatrix{T},
    A2::AbstractMatrix{T},
    A3::AbstractMatrix{T},
    A4::AbstractMatrix{T};
    shift::T = zero(T)
) where T
    # Az = A1 - z * A2 - conj(z) * A3 + abs2(z) * A4

    # check dimensions
    @assert size(A1) == size(A2) == size(A3) == size(A4) == size(Az)

    m, n = size(A1)
    zc = conj(z)
    zz = abs2(z) + shift
    @inbounds @simd for j = 1:n
        for i = 1:m
            Az[i, j] = A1[i, j] - z*A2[i, j] - zc*A3[i, j] + zz*A4[i, j]
        end
    end
end

"""
    check_rr_dim(lambda, gap)

Return the first Ritz dimension where adjacent eigenvalues have a clear gap.
"""
function check_rr_dim(
    λ::AbstractVector{T},
    gap::Float64
) where T
    # find i, that |λ[i] - λ[i+1]| > τ
    n = length(λ)
    @inbounds for i = 1:(n-1)
        if abs(λ[i] - λ[i+1]) > gap
            return i
        end
    end
    return 0
end

"""
    find_rr_dim!(Hz, G, lambda, r, tau, n_init = 2r, n_max = -1)

Increase the projected eigenproblem size until a Ritz-value gap is found.
"""
function find_rr_dim!(
    Hz::AbstractMatrix{T},
    G::AbstractMatrix{T},
    λ::AbstractVector{RT},
    r::Int,
    τ::Float64,
    n_init::Int = 2*r,
    n_max::Int = -1
) where {T, RT}
    n_H = size(Hz, 1)
    n_init = min(n_init, n_H)
    n_max = n_max > 0 ? min(n_max, n_H) : n_H

    n = n_init
    r_ = 0
    @no_escape begin
        Hz0 = @alloc(T, n_H, n_H)
        copyto!(Hz0, Hz)

        # compute maximum eigenvalue of Hz and gap
        rls_eigen!(Hz, λ, G; range = 'I', il = n_H, iu = n_H)
        λ_max = λ[1]
        gap = λ_max / τ * eps(RT)

        while n <= n_max
            # rls_eigen! (LAPACK syevr/heevr) overwrites input matrix,
            # so restore Hz each iteration before eigen solve.
            copyto!(Hz, Hz0)
            rls_eigen!(Hz, λ, G; range = 'I', il = 1, iu = n)
            # @views println("[RR_LAMBDA] n=", n, " lambda=", λ[1:n])
            @views r_ = check_rr_dim(λ[1:n], gap)
            if r_ > 0
                break
            else 
                n = n + 2*r
            end
        end
    end
    if n_max < n_H && r_ == 0
        @warn "Failed to find a clear gap in Ritz values within n_max = $n_max. Consider increasing n_max or adjusting τ."
    end
    return r_ > 0 ? max(r_, r) : n_max
end

# Shared implementation for implicit and explicit recycled SVD Rayleigh-Ritz paths.
function _fastSvdRaylRitz_impl!(
    ::Val{USE_EXPLICIT_MS},
    recySp::RecycledSpace{T},
    z::T,
    X_recy::AbstractMatrix{T},
    W_recy::AbstractMatrix{T},
    σ_recy::AbstractVector{RT},
    M,
    S;
    shift::T,
    τ::Float64
) where {USE_EXPLICIT_MS, T, RT}
    @unpack p, m, n, r, pts, V, L, H1, H2, H3, J1, J2, J3, J4, J5, J6 = recySp

    pr = p*r
    pr == 0 && return 0

    if USE_EXPLICIT_MS
        size(M) == (m, n) || throw(DimensionMismatch("size(M) = $(size(M)) must be ($m, $n)"))
        size(S) == (m, n) || throw(DimensionMismatch("size(S) = $(size(S)) must be ($m, $n)"))
    elseif !recySp.implict_recycle
        throw(ArgumentError("fastSvdRaylRitz!: recySp.implict_recycle=false requires explicit M and S"))
    end

    # compute H
    @views H1_ = H1[1:pr, 1:pr]
    @views H2_ = H2[1:pr, 1:pr]
    @views H3_ = H3[1:pr, 1:pr]
    
    r_ = 0
    @no_escape begin
        Hz = @alloc(T, pr, pr)
        G  = @alloc(T, pr, pr)
        λ  = @alloc(real(T), pr)   
        affine_conbine!(Hz, z, H1_, H2_, H2_', H3_; shift = shift)
        r_ = find_rr_dim!(Hz, G, λ, r, τ)

        # get CX_
        @views G1 = G[:, 1:r_]
        if USE_EXPLICIT_MS
            # explicit branch:
            # 1) X_tmp = V * G1
            # 2) MX_tmp = (M - zS) * X_tmp
            # 3) QR/SVD on MX_tmp, then rotate X_tmp
            X_tmp = @alloc(T, n, r_)
            @views mul!(X_tmp, V[:, 1:pr], G1)

            MX_tmp = @alloc(T, m, r_)
            SX_tmp = @alloc(T, m, r_)
            mul!(MX_tmp, M, X_tmp)
            mul!(SX_tmp, S, X_tmp)
            @views @. MX_tmp = MX_tmp - z * SX_tmp

            σ = @alloc(real(T), r_)
            R_ = @alloc(T, r_, r_)
            G_ = @alloc(T, r_, r_)
            rls_qr!(MX_tmp, R_)
            rls_svd!(R_, σ, G_; return_V = true, ascending = true)
            @views G2 = G_[:, 1:r]
            @views σ_recy[1:r] .= σ[1:r]

            mul!(X_recy, X_tmp, G2)

            # W = (M-zS)' * ((M-zS)X_recy) - X_recy * Σ^2
            MX = @alloc(T, m, r)
            SX = @alloc(T, m, r)
            mul!(MX, M, X_recy)
            mul!(SX, S, X_recy)
            @views @. MX = MX - z * SX

            Mtx = @alloc(T, n, r)
            Stx = @alloc(T, n, r)
            mul!(Mtx, M', MX)
            mul!(Stx, S', MX)
            @views @. W_recy = Mtx - conj(z) * Stx
            @inbounds for j in 1:r
                s2 = σ_recy[j] * σ_recy[j]
                @simd for i in axes(W_recy, 1)
                    W_recy[i, j] -= X_recy[i, j] * s2
                end
            end
        else
            CzX  = @alloc(T, m, pr)
            CzX_ = @alloc(T, m, r_) 
            @views @. CzX = J5[:, 1:pr] - z * J6[:, 1:pr]
            mul!(CzX_, CzX, G1)

            # perform Svd-Rayleigh-Ritz on CzX_
            σ  = @alloc(real(T), r_)
            R_ = @alloc(T, r_, r_)
            G_ = @alloc(T, r_, r_)
            rls_qr!(CzX_, R_)
            rls_svd!(R_, σ, G_; return_V = true, ascending = true)
            @views G2 = G_[:, 1:r]
            @views σ_recy[1:r] .= σ[1:r]

            G_total = @alloc(T, pr, r)
            if r_ < pr
                # we use the two-level Ritz vectors
                mul!(G_total, G1, G2)
            else 
                # we use the original Ritz vectors
                copyto!(G_total, G2)
            end
            work = @alloc(T, n, pr)
            @views affine_conbine!(work, z, J1[:, 1:pr], J2[:, 1:pr], J3[:, 1:pr], J4[:, 1:pr])

            # get X_recy
            @views mul!(X_recy, V[:, 1:pr], G_total)

            # get W_recy
            mul!(W_recy, work, G_total)
            @inbounds for j in axes(W_recy, 2)
                s2 = σ_recy[j] * σ_recy[j]
                @simd for i in axes(W_recy, 1)
                    W_recy[i, j] -= X_recy[i, j] * s2
                end
            end
        end
    end
    return r_
end

"""
    fastSvdRaylRitz_implicit!(recySp, z, X_recy, W_recy, sigma_recy; shift, tau)

Run recycled SVD Rayleigh-Ritz using cached `J` blocks in `recySp`.
"""
function fastSvdRaylRitz_implicit!(
    recySp::RecycledSpace{T},
    z::T,
    X_recy::AbstractMatrix{T},
    W_recy::AbstractMatrix{T},
    σ_recy::AbstractVector{RT};
    shift::T,
    τ::Float64
) where {T, RT}
    return _fastSvdRaylRitz_impl!(
        Val(false),
        recySp,
        z,
        X_recy,
        W_recy,
        σ_recy,
        nothing,
        nothing;
        shift = shift,
        τ = τ
    )
end

"""
    fastSvdRaylRitz_explicit!(recySp, z, X_recy, W_recy, sigma_recy, M, S; shift, tau)

Run recycled SVD Rayleigh-Ritz by explicitly applying `M - zS`.
"""
function fastSvdRaylRitz_explicit!(
    recySp::RecycledSpace{T},
    z::T,
    X_recy::AbstractMatrix{T},
    W_recy::AbstractMatrix{T},
    σ_recy::AbstractVector{RT},
    M::AbstractMatrix{T},
    S::AbstractMatrix{T};
    shift::T,
    τ::Float64
) where {T, RT}
    return _fastSvdRaylRitz_impl!(
        Val(true),
        recySp,
        z,
        X_recy,
        W_recy,
        σ_recy,
        M,
        S;
        shift = shift,
        τ = τ
    )
end

"""
    fastSvdRaylRitz!(recySp, z, X_recy, W_recy, sigma_recy; shift, tau, M = nothing, S = nothing)

Dispatch to the implicit cached path or the explicit operator path.
"""
function fastSvdRaylRitz!(
    recySp::RecycledSpace{T},
    z::T,
    X_recy::AbstractMatrix{T},
    W_recy::AbstractMatrix{T},
    σ_recy::AbstractVector{RT};
    shift::T,
    τ::Float64,
    M::Union{Nothing, AbstractMatrix{T}} = nothing,
    S::Union{Nothing, AbstractMatrix{T}} = nothing
) where {T, RT}
    if M === nothing
        S === nothing || throw(ArgumentError("M and S must be both provided or both nothing"))
        return fastSvdRaylRitz_implicit!(
            recySp,
            z,
            X_recy,
            W_recy,
            σ_recy;
            shift = shift,
            τ = τ
        )
    else
        S === nothing && throw(ArgumentError("M and S must be both provided or both nothing"))
        return fastSvdRaylRitz_explicit!(
            recySp,
            z,
            X_recy,
            W_recy,
            σ_recy,
            M,
            S;
            shift = shift,
            τ = τ
        )
    end
end

"""
    stanardSvdRaylRitz!(recySp, z, X_recy, W_recy, sigma_recy; shift)

Reference SVD Rayleigh-Ritz implementation on the full recycled basis.
"""
function stanardSvdRaylRitz!(
    recySp::RecycledSpace{T},
    z::T,
    X_recy::AbstractMatrix{T},
    W_recy::AbstractMatrix{T},
    σ_recy::AbstractVector{RT};
    shift::T
) where {T, RT}
    @unpack p, m, n, r, pts, V, L, H1, H2, H3, J1, J2, J3, J4, J5, J6 = recySp

    pr = p * r
    pr == 0 && return 0

    @no_escape begin
        r_eff = min(r, pr)

        # direct RR-SVD on (M - zS)V
        CzV = @alloc(T, m, pr)
        @views @. CzV = J5[:, 1:pr] - z * J6[:, 1:pr]

        R_ = @alloc(T, pr, pr)
        rls_qr!(CzV, R_)

        σ = @alloc(real(T), pr)
        G = @alloc(T, pr, pr)
        rls_svd!(R_, σ, G; return_V = true, ascending = true)

        @views G_ = G[:, 1:r_eff]
        @views σ_recy[1:r_eff] .= σ[1:r_eff]

        # X_recy = V * G_
        @views mul!(X_recy[:, 1:r_eff], V[:, 1:pr], G_)

        # W_recy = ((M-zS)'(M-zS)V)G_ - X_recy * Σ^2
        work = @alloc(T, n, pr)
        @views affine_conbine!(work, z, J1[:, 1:pr], J2[:, 1:pr], J3[:, 1:pr], J4[:, 1:pr]; shift = shift)
        @views mul!(W_recy[:, 1:r_eff], work, G_)

        @inbounds for j in 1:r_eff
            s2 = σ_recy[j] * σ_recy[j]
            @simd for i in axes(W_recy, 1)
                W_recy[i, j] -= X_recy[i, j] * s2
            end
        end
    end

    return pr
end

"""
    fastRaylRitz!(recySp, z, X_recy, E_recy, W_recy, lambda_recy)

Run Rayleigh-Ritz on the cached normal-equation projection in `recySp`.
"""
function fastRaylRitz!(
    recySp::RecycledSpace{T},
    z::T,
    X_recy::AbstractMatrix{T},
    E_recy::AbstractMatrix{T},
    W_recy::AbstractMatrix{T},
    λ_recy::AbstractVector{RT}
) where {T, RT}
    recySp.implict_recycle || throw(ArgumentError("fastRaylRitz! requires recySp.implict_recycle=true"))

    @unpack p, m, n, r, pts, V, L, H1, H2, H3, J1, J2, J3, J4, J5, J6 = recySp

    pr = p*r
    pr == 0 && return nothing
    # compute H
    @views H1_ = H1[1:pr, 1:pr]
    @views H2_ = H2[1:pr, 1:pr]
    @views H3_ = H3[1:pr, 1:pr]
    
    r_ = 0
    @no_escape begin
        Hz = @alloc(T, pr, pr)
        G  = @alloc(T, pr, pr)
        λ  = @alloc(real(T), pr)   
        affine_conbine!(Hz, z, H1_, H2_, H2_', H3_)
        iu = min(2r, pr)
        rls_eigen!(Hz, λ, G; range = 'I', il = 1, iu = iu)
        @views λ_recy[1:r] .= λ[1:r]

        @views G_ = G[:, 1:r]
        @views mul!(X_recy, V[:, 1:pr], G_)
        @views mul!(E_recy, J6[1:n, 1:pr], G_)

        work = @alloc(T, n, pr)
        @views affine_conbine!(work, z, J1[:, 1:pr], J2[:, 1:pr], J3[:, 1:pr], J4[:, 1:pr])
        @views mul!(W_recy, work, G_)

        @inbounds for j in axes(W_recy, 2)
            s2 = λ_recy[j]
            @simd for i in axes(W_recy, 1)
                W_recy[i, j] -= X_recy[i, j] * s2
            end
        end

    end
end
