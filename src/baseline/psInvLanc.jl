using KrylovKit

@inline _psInvLanc_time_ns() = time_ns()
@inline _psInvLanc_elapsed_ns(t_start::UInt64) = time_ns() - t_start
@inline _psInvLanc_ns_to_s(t::UInt64) = Float64(t) / 1e9

Base.@kwdef struct PsInvLancControl
    tol::Float64 = 1e-4
    maxiter::Int = 800
    krylovdim::Int = 20
    ordering::Int = 0
    continue_::Bool = false
    verbose::Int = 1
end

@inline function _largest_sv_krylov(
    A::AbstractMatrix{T},
    x0::AbstractVector{T};
    tol::Float64,
    maxiter::Int,
    krylovdim::Int
) where {T}
    size(A, 2) == length(x0) || throw(DimensionMismatch("x0 length mismatch with matrix column size"))
    σ, ~, V, info = svdsolve(
        A,
        x0,
        1,
        :LR;
        tol = tol,
        maxiter = maxiter,
        eager = true,
        krylovdim = krylovdim
    )
    return real(σ[1]), V[1], info
end

@inline function _print_invlan_step!(
    io::IO,
    i::Int,
    nZ::Int,
    z,
    mv::Int,
    σmin::Real
)
    @printf(
        io,
        "[%4d/%4d] z=%9.4f%+9.4fim | invlan: mv=%4d σ_min=%11.4e\n",
        i,
        nZ,
        real(z),
        imag(z),
        mv,
        σmin
    )
    @printf(io, "%s\n", repeat("-", 100))
    flush(io)
    return nothing
end

function _print_psInvLanc_startup!(
    io::IO,
    rec::Recursion
)
    n_pts = rec.n_x * rec.n_y
    @printf(
        io,
        "[psInvLanc] rec: x_range=[%.6f, %.6f], y_range=[%.6f, %.6f], n_x=%d, n_y=%d, k=%d, n_pts=%d\n",
        rec.x_st,
        rec.x_ed,
        rec.y_st,
        rec.y_ed,
        rec.n_x,
        rec.n_y,
        rec.k,
        n_pts
    )
    flush(io)
    return nothing
end

"""
    psInvLanc(C, rec; control = PsInvLancControl(), io = stdout)

Evaluate the smallest singular value over a recursion grid using inverse Lanczos
on the shifted preprocessed problem. Returns `(sigma_min, stats)`.
"""
function psInvLanc(
    C::AbstractMatrix{T},
    rec::Recursion;
    control::PsInvLancControl = PsInvLancControl(),
    io::IO = stdout
) where {T<:FloatOrComplex}
    m, n = size(C)
    if C isa SparseMatrixCSC
        m == n || throw(DimensionMismatch("psInvLanc only supports square sparse matrices, got size $(size(C))"))
    else
        m >= n || throw(DimensionMismatch("psInvLanc only supports dense matrices with m >= n, got size $(size(C))"))
    end

    C_work = C isa SparseMatrixCSC ? C : Matrix(C)
    preprocess_t_start = _psInvLanc_time_ns()
    M, S = pre_process(C_work)
    preprocess_time_ns = _psInvLanc_elapsed_ns(preprocess_t_start)

    if control.verbose >= 1
        _print_psInvLanc_startup!(io, rec)
    end

    nZ = length(rec.Z)
    σ_min = zeros(real(T), nZ)
    mv = zeros(Int, nZ)
    qr_time_ns = zero(UInt64)
    invlan_time_ns = zero(UInt64)
    x0 = randn(T, n)

    use_sparse_shift = M isa SparseMatrixCSC
    use_dense_diag_shift = M isa UpperTriangular{T, Matrix{T}} && S isa Diagonal{T, Vector{T}}
    Rz = nothing
    Mz = nothing
    mz_map_M = Int[]
    mz_map_S = Int[]
    if use_sparse_shift
        Msp = M::SparseMatrixCSC{T}
        Ssp = S::SparseMatrixCSC{T}
        Mz, mz_map_M, mz_map_S = _build_sparse_shift_workspace(Msp, Ssp)
    else
        if use_dense_diag_shift
            Md = M::UpperTriangular{T, Matrix{T}}
            Rz = UpperTriangular(copy(parent(Md)))
        else
            Mz = zeros(T, size(M))
        end
    end

    for i = 1:nZ
        z = rec.Z[i]
        if !control.continue_ && i > 1
            randn!(x0)
        end

        qr_time_ns_i = zero(UInt64)
        Rcurr = if use_sparse_shift
            Mz_sp = Mz::SparseMatrixCSC{T}
            M_sp = M::SparseMatrixCSC{T}
            S_sp = S::SparseMatrixCSC{T}
            _update_sparse_shift_matrix!(
                Mz_sp,
                M_sp,
                S_sp,
                mz_map_M,
                mz_map_S,
                z
            )
            qr_t_start = _psInvLanc_time_ns()
            F = qr(Mz_sp; ordering = 0)
            qr_time_ns_i = _psInvLanc_elapsed_ns(qr_t_start)
            UpperTriangular(F.R)
        else
            if use_dense_diag_shift
                Rz_dense = Rz::UpperTriangular{T, Matrix{T}}
                M_dense = M::UpperTriangular{T, Matrix{T}}
                S_dense = S::Diagonal{T, Vector{T}}
                get_R_factor!(
                    Rz_dense,
                    M_dense,
                    S_dense,
                    z
                )
                Rz_dense
            else
                Mz_dense = Mz::Matrix{T}
                M_dense = M::AbstractMatrix{T}
                S_dense = S::AbstractMatrix{T}
                get_R_factor!(
                    Mz_dense,
                    M_dense,
                    S_dense,
                    z
                )
            end
        end
        qr_time_ns += qr_time_ns_i

        invR = InvMat(Rcurr)
        invlan_t_start = _psInvLanc_time_ns()
        σ_inv_max, vmax, info = _largest_sv_krylov(
            invR,
            x0;
            tol = control.tol,
            maxiter = control.maxiter,
            krylovdim = control.krylovdim
        )
        invlan_time_ns += _psInvLanc_elapsed_ns(invlan_t_start)
        if control.continue_
            copyto!(x0, vmax)
        end
        mv[i] = info.numops

        if isfinite(σ_inv_max) && σ_inv_max > 0
            σ_min[i] = inv(σ_inv_max)
        else
            σ_min[i] = zero(real(T))
            control.verbose >= 1 && @printf(
                io,
                "[psInvLanc] non-finite σ_inv_max at i=%d z=%9.4f%+9.4fim\n",
                i,
                real(z),
                imag(z)
            )
        end

        if control.verbose >= 1
            _print_invlan_step!(
                io,
                i,
                nZ,
                z,
                info.numops,
                σ_min[i]
            )
        end

        if control.verbose >= 2
            @printf(
                io,
                "[invlan-info] %s\n",
                string(info)
            )
            @printf(
                io,
                "[%4d/%4d] z=%9.4f%+9.4fim | invlan: mv=%4d σ_min=%11.4e\n",
                i,
                nZ,
                real(z),
                imag(z),
                info.numops,
                σ_min[i]
            )
            @printf(io, "%s\n", repeat("-", 100))
        end
    end

    stats = (
        mv = mv,
        n_pts = nZ,
        pts = copy(rec.Z),
        preprocess_time_ns = preprocess_time_ns,
        qr_time_ns = qr_time_ns,
        invlan_time_ns = invlan_time_ns
    )
    if control.verbose >= 1
        @printf(io, "Total preprocess time for singular value problem: %.6f seconds\n", _psInvLanc_ns_to_s(preprocess_time_ns))
        if qr_time_ns > 0
            @printf(io, "Total QR time for singular value problem: %.6f seconds\n", _psInvLanc_ns_to_s(qr_time_ns))
        end
        @printf(io, "Total inverse Lanczos time for singular value problem: %.6f seconds\n", _psInvLanc_ns_to_s(invlan_time_ns))
        println(io, "Total R vec apps for singular value problem: ", sum(mv))
        flush(io)
    end
    return σ_min, stats
end
