# Allocation-free LAPACK wrappers.
# Bumper provides temporary workspace; metaprogramming emits type-specialized methods.

using LinearAlgebra
using LinearAlgebra: BlasInt, chkstride1, require_one_based_indexing, AbstractRotation, AdjointRotation
using LinearAlgebra.BLAS: @blasfunc, libblastrampoline
using LinearAlgebra.LAPACK: chklapackerror
using Bumper
using BenchmarkTools

import LinearAlgebra: mul!, rmul!, lmul!, adjoint
import Base: size, copy

# In-place Cholesky wrapper for consistency with rls_* APIs.
@inline function rls_cholesky!(A::AbstractMatrix{T}; uplo::Symbol = :U, check::Bool = true) where {T}
    require_one_based_indexing(A)
    chkstride1(A)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("Cholesky requires square matrix, got $(size(A))"))
    return cholesky!(Hermitian(A, uplo); check = check)
end

"""
    BlockUnitaryRot{T} <: AbstractRotation{T}

Block rotation: apply a 2r-by-2r unitary matrix `U` to two block rows or columns of size `r`.
The first block index is `i1`, and the second is `i2` (1-based).
"""


struct BlockUnitaryRot{T, M<:AbstractMatrix{T}} <: AbstractRotation{T}
    i1::Int
    i2::Int
    r::Int
    U::M
end


function BlockUnitaryRot(i1::Integer, i2::Integer,
                         r::Integer, U::AbstractMatrix{T}) where {T}
    @assert size(U,1) == size(U,2) "U must be square"
    @assert size(U,1) == 2*r "size(U,1) must be 2*r"
    BlockUnitaryRot{T}(Int(i1), Int(i2), Int(r), U)
end

BlockUnitaryRot{T}(G::BlockUnitaryRot{T}) where {T} = G
BlockUnitaryRot{T}(G::BlockUnitaryRot)    where {T} =
    BlockUnitaryRot(G.i1, G.i2, G.r, G.U)

AbstractRotation{T}(G::BlockUnitaryRot)   where {T} = BlockUnitaryRot{T}(G)

adjoint(G::BlockUnitaryRot{T}) where {T} =
    BlockUnitaryRot(G.i1, G.i2, G.r, adjoint(G.U))

# Expose a size for debugging and display.
size(G::BlockUnitaryRot) = (2*G.r, 2*G.r)

# BlockUnitaryRot left and right multiplication.

@inline function lmul!(G::BlockUnitaryRot{T}, A::AbstractMatrix{T},
    temp1::AbstractMatrix{T}, temp2::AbstractMatrix{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    i1, i2, r = G.i1, G.i2, G.r
    @assert size(temp1) == (r, n) "temp1 has wrong size"
    @assert size(temp2) == (r, n) "temp2 has wrong size"
    r1 = (i1-1)*r + 1 : i1*r
    r2 = (i2-1)*r + 1 : i2*r
    if last(r2) > m
        throw(DimensionMismatch("row block indices for BlockUnitaryRot are outside the matrix"))
    end

    # Split U into block rows and apply it to the matching block rows of A.
    @views U11 = G.U[1:r, 1:r]
    @views U12 = G.U[1:r, r+1:2r]
    @views U21 = G.U[r+1:2r, 1:r]
    @views U22 = G.U[r+1:2r, r+1:2r]

    @views A1 = A[r1, :]    # r x n
    @views A2 = A[r2, :]    # r x n

    # Compute temp1 = U11*A1 + U12*A2.
    @views mul!(temp1, U11, A1)      # r x n
    @views mul!(temp1, U12, A2, one(T), one(T))  

    # Compute temp2 = U21*A1 + U22*A2.
    @views mul!(temp2, U21, A1)      # r x n
    @views mul!(temp2, U22, A2, one(T), one(T))

    copyto!(A1, temp1)
    copyto!(A2, temp2)
end

@inline function lmul!(G::BlockUnitaryRot{T}, A::AbstractMatrix{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    r = G.r
    @no_escape begin
        temp1 = @alloc(T, r, n)
        temp2 = @alloc(T, r, n)
        lmul!(G, A, temp1, temp2)
    end
end


@inline function rmul!(A::AbstractMatrix{T}, G::BlockUnitaryRot{T}, temp1::AbstractMatrix{T}, temp2::AbstractMatrix{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    i1, i2, r = G.i1, G.i2, G.r
    @assert size(temp1) == (m, r) "temp1 has wrong size"
    @assert size(temp2) == (m, r) "temp2 has wrong size"
    c1 = (i1-1)*r + 1 : i1*r
    c2 = (i2-1)*r + 1 : i2*r
    if last(c2) > n
        throw(DimensionMismatch("column block indices for BlockUnitaryRot are outside the matrix"))
    end

    # Split U into block columns and apply it to the matching block columns of A.
    @views U11 = G.U[1:r, 1:r]
    @views U12 = G.U[1:r, r+1:2r]
    @views U21 = G.U[r+1:2r, 1:r]
    @views U22 = G.U[r+1:2r, r+1:2r]

    @views A1 = A[:, c1]    # m x bs
    @views A2 = A[:, c2]    # m x bs

    # Compute temp1 = A1*U11 + A2*U21.
    @views mul!(temp1, A1, U11)      # m x bs
    @views mul!(temp1, A2, U21, one(T), one(T))

    # Compute temp2 = A1*U12 + A2*U22.
    @views mul!(temp2, A1, U12)      # m x bs
    @views mul!(temp2, A2, U22, one(T), one(T))

    copyto!(A1, temp1)
    copyto!(A2, temp2)
end

@inline function rmul!(A::AbstractMatrix{T}, G::BlockUnitaryRot{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    r = G.r
    @no_escape begin
        temp1 = @alloc(T, m, r)
        temp2 = @alloc(T, m, r)
        rmul!(A, G, temp1, temp2)
    end
end

"""
    make_block_unitary_rot(A, B, i1, i2)

Given two r-by-r blocks A and B, build a BlockUnitaryRot such that

    Y = [A; B]  (2r x r)
    G = make_block_unitary_rot(...)
    G * Y ~= [R; 0]

In floating-point arithmetic, this zeros the lower block B.

- A, B :: AbstractMatrix{T}, size = (r, r)
- i1, i2 :: Integer, block indices in the parent matrix (1-based)
"""
function rls_block_unitary_rot!(
    A::AbstractMatrix{T}, B::AbstractMatrix{T},
    i1::Integer, i2::Integer,
    U::AbstractMatrix{T}
) where {T}
    # check square and same size
    require_one_based_indexing(A, B)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("A must be square"))
    size(A) == size(B) || throw(DimensionMismatch("A and B must have the same size"))

    r = size(A, 1)

    @no_escape begin
        # Y = [A; B] : 2r x r
        Y = @alloc(T, 2r, r)
        Q = @alloc(T, 2r, 2r)
        @views copyto!(Y[1:r, :], A)
        @views copyto!(Y[r+1:2r, :], B)

        # Local QR: Y ~= Q * R, where Q is a 2r-by-2r unitary matrix.
        rls_qr!(Y, Q, nothing)  # Y is overwritten with Q.

        # Set U = Q'.
        U .= Q'
        nothing
    end
    return BlockUnitaryRot(i1, i2, r, U)
end





#=
rls_svd!(A, U, S, Vt; return_V=false) -> minmn

Allocation-free SVD wrapper using Bumper workspace (job='S').
Writes results to U, S, and Vt, and returns the active dimension minmn.

Arguments:
- A: input matrix, overwritten in place, m x n
- U: preallocated U matrix, m x minmn, or nothing for temporary storage
- S: preallocated singular-value vector, length >= min(m,n)
- Vt: preallocated Vt matrix, minmn x n; with return_V=true it is converted in place to V
- return_V: convert Vt to V in place, using adjoint for complex values; supports only square m >= n cases

Returns:
- minmn: number of active singular values
=#

@inline function _inplace_adjoint_square!(A::AbstractMatrix{T}) where {T}
    n1, n2 = size(A)
    n1 == n2 || throw(ArgumentError("in-place adjoint only supports square matrices, got ($n1, $n2)"))
    @inbounds for i = 1:n1
        A[i, i] = conj(A[i, i])
        for j = i+1:n1
            tmp = A[i, j]
            A[i, j] = conj(A[j, i])
            A[j, i] = conj(tmp)
        end
    end
    return nothing
end

@inline function _reverse_vec!(x::AbstractVector)
    i, j = firstindex(x), lastindex(x)
    @inbounds while i < j
        x[i], x[j] = x[j], x[i]
        i += 1
        j -= 1
    end
    return nothing
end

@inline function _reverse_cols!(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    j1, j2 = 1, n
    @inbounds while j1 < j2
        for i = 1:m
            A[i, j1], A[i, j2] = A[i, j2], A[i, j1]
        end
        j1 += 1
        j2 -= 1
    end
    return nothing
end

# Generate Float32 and Float64 real-valued methods.
for (elty, gesdd) in ((:Float32, :sgesdd_),
                       (:Float64, :dgesdd_))
    @eval begin
        function rls_svd!(
            A::AbstractMatrix{$elty},
            U::AbstractMatrix{$elty},
            S::AbstractVector{$elty},
            Vt::AbstractMatrix{$elty};
            return_V::Bool = false,
            ascending::Bool = false
        )
            require_one_based_indexing(A, U, Vt)
            chkstride1(A, U, Vt)
            
            m, n = size(A)
            minmn = min(m, n)
            job = UInt8('S')
            
            # Check dimensions.
            size(U, 1) >= m || throw(DimensionMismatch("U has too few rows: $(size(U,1)) < $m"))
            size(U, 2) >= minmn || throw(DimensionMismatch("U has too few columns: $(size(U,2)) < $minmn"))
            size(Vt, 1) >= minmn || throw(DimensionMismatch("Vt has too few rows: $(size(Vt,1)) < $minmn"))
            size(Vt, 2) >= n || throw(DimensionMismatch("Vt has too few columns: $(size(Vt,2)) < $n"))
            length(S) >= minmn || throw(DimensionMismatch("S is too short: $(length(S)) < $minmn"))
            
            ldu = max(1, stride(U, 2))
            ldvt = max(1, stride(Vt, 2))
            
            @no_escape begin
                # First call: query optimal lwork.
                work_query = @alloc($elty, 1)
                iwork_ptr = @alloc(BlasInt, 8*minmn)
                info_ptr = @alloc(BlasInt, 1)
                
                lwork_query = BlasInt(-1)
                
                ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                      job, m, n, pointer(A), max(1, stride(A, 2)), 
                      pointer(S), pointer(U), ldu, 
                      pointer(Vt), ldvt, 
                      pointer(work_query), lwork_query, pointer(iwork_ptr), pointer(info_ptr), 1)
                
                # Read the optimal lwork.
                lwork = round(BlasInt, nextfloat(unsafe_load(pointer(work_query))))
                
                # Allocate optimal workspace.
                work_ptr = @alloc($elty, lwork)
                
                # Second call: compute the SVD.
                ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                      job, m, n, pointer(A), max(1, stride(A, 2)), 
                      pointer(S), pointer(U), ldu, 
                      pointer(Vt), ldvt, 
                      pointer(work_ptr), lwork, pointer(iwork_ptr), pointer(info_ptr), 1)
                
                info_val = unsafe_load(pointer(info_ptr))
                chklapackerror(info_val)
            end

            if return_V
                m >= n || throw(ArgumentError("return_V=true only supports m >= n, got m=$m, n=$n"))
                _inplace_adjoint_square!(view(Vt, 1:minmn, 1:minmn))
            end
            if ascending
                _reverse_vec!(view(S, 1:minmn))
                if return_V
                    _reverse_cols!(view(Vt, 1:minmn, 1:minmn))
                end
            end
            
            return minmn
        end
    end
end

# Generate ComplexF32 and ComplexF64 methods.
for (elty, relty, gesdd) in ((:ComplexF32, :Float32, :cgesdd_),
                              (:ComplexF64, :Float64, :zgesdd_))
    @eval begin
        function rls_svd!(
            A::AbstractMatrix{$elty},
            U::AbstractMatrix{$elty},
            S::AbstractVector{$relty},
            Vt::AbstractMatrix{$elty};
            return_V::Bool = false,
            ascending::Bool = false
        )
            require_one_based_indexing(A, U, Vt)
            chkstride1(A, U, Vt)
            
            m, n = size(A)
            minmn = min(m, n)
            job = UInt8('S')
            
            # Check dimensions.
            size(U, 1) >= m || throw(DimensionMismatch("U has too few rows: $(size(U,1)) < $m"))
            size(U, 2) >= minmn || throw(DimensionMismatch("U has too few columns: $(size(U,2)) < $minmn"))
            size(Vt, 1) >= minmn || throw(DimensionMismatch("Vt has too few rows: $(size(Vt,1)) < $minmn"))
            size(Vt, 2) >= n || throw(DimensionMismatch("Vt has too few columns: $(size(Vt,2)) < $n"))
            length(S) >= minmn || throw(DimensionMismatch("S is too short: $(length(S)) < $minmn"))
            
            ldu = max(1, stride(U, 2))
            ldvt = max(1, stride(Vt, 2))
            rwork_size = minmn * max(5*minmn + 7, 2*max(m, n) + 2*minmn + 1)
            
            @no_escape begin
                # First call: query optimal lwork.
                work_query = @alloc($elty, 1)
                iwork_ptr = @alloc(BlasInt, 8*minmn)
                rwork_ptr = @alloc($relty, rwork_size)
                info_ptr = @alloc(BlasInt, 1)
                
                lwork_query = BlasInt(-1)
                
                ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                      job, m, n, pointer(A), max(1, stride(A, 2)), 
                      pointer(S), pointer(U), ldu, 
                      pointer(Vt), ldvt, 
                      pointer(work_query), lwork_query, pointer(rwork_ptr), pointer(iwork_ptr), pointer(info_ptr), 1)
                
                # Read the optimal lwork.
                lwork = round(BlasInt, nextfloat(real(unsafe_load(pointer(work_query)))))
                
                # Allocate optimal workspace.
                work_ptr = @alloc($elty, lwork)
                
                # Second call: compute the SVD.
                ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                      job, m, n, pointer(A), max(1, stride(A, 2)), 
                      pointer(S), pointer(U), ldu, 
                      pointer(Vt), ldvt, 
                      pointer(work_ptr), lwork, pointer(rwork_ptr), pointer(iwork_ptr), pointer(info_ptr), 1)
                
                info_val = unsafe_load(pointer(info_ptr))
                chklapackerror(info_val)
            end

            if return_V
                m >= n || throw(ArgumentError("return_V=true only supports m >= n, got m=$m, n=$n"))
                _inplace_adjoint_square!(view(Vt, 1:minmn, 1:minmn))
            end
            if ascending
                _reverse_vec!(view(S, 1:minmn))
                if return_V
                    _reverse_cols!(view(Vt, 1:minmn, 1:minmn))
                end
            end
            
            return minmn
        end
    end
end

# Convenience interface that only requires S and Vt; U uses temporary Bumper storage.
function rls_svd!(
    A::AbstractMatrix{T},
    S::AbstractVector,
    Vt::AbstractMatrix{T};
    return_V::Bool = false,
    ascending::Bool = false
) where T
    m, n = size(A)
    minmn = min(m, n)
    @no_escape begin
        U_tmp = @alloc(T, m, minmn)
        rls_svd!(A, U_tmp, S, Vt; return_V=return_V, ascending=ascending)
    end
    return minmn
end

# Convenience interface that only requires S; U and Vt use temporary Bumper storage.
function rls_svd!(A::AbstractMatrix{T}, S::AbstractVector) where T
    m, n = size(A)
    minmn = min(m, n)
    @no_escape begin
        U_tmp = @alloc(T, m, minmn)
        Vt_tmp = @alloc(T, minmn, n)
        rls_svd!(A, U_tmp, S, Vt_tmp)
    end
    return minmn
end

#=
rls_qr!(V, Q, R; type = :lapack) -> nothing

Allocation-free QR factorization using Bumper workspace.
V is overwritten with the first n columns of Q, Q stores the full orthogonal matrix,
and R stores the upper-triangular factor.

Arguments:
- V: input matrix m x n, overwritten with the first n columns of Q; requires m >= n
- Q: preallocated orthogonal matrix output m x m, or nothing to form Q in V
- R: preallocated upper-triangular output n x n, or nothing to skip R extraction

Returns:
- nothing; results are stored in V, Q, and R in place

Keywords:
- `type = :lapack` uses the Householder/LAPACK implementation
- `type = :shiftchol` uses shifted Cholesky QR and only supports thin QR (`Q === nothing`)
=#

# Generate QR factorization methods; real types use orgqr and complex types use ungqr.
for (elty, geqrf, xgqr) in ((:Float32,    :sgeqrf_, :sorgqr_),
                             (:Float64,    :dgeqrf_, :dorgqr_),
                             (:ComplexF32, :cgeqrf_, :cungqr_),
                             (:ComplexF64, :zgeqrf_, :zungqr_))
    @eval begin
        function _rls_qr_lapack_impl!(
            V::AbstractMatrix{$elty},
            Q::Union{Nothing, AbstractMatrix{$elty}},
            R::Union{Nothing, AbstractMatrix{$elty}}
        )
            require_one_based_indexing(V)
            chkstride1(V)
            
            m, n = size(V)
            @assert m >= n "QR factorization requires m >= n, got m=$m, n=$n"

            # Check dimensions.
            if Q !== nothing
                require_one_based_indexing(Q)
                chkstride1(Q)
                size(Q, 1) >= m || throw(DimensionMismatch("Q has too few rows: $(size(Q,1)) < $m"))
                size(Q, 2) >= m || throw(DimensionMismatch("Q has too few columns: $(size(Q,2)) < $m"))
            end


            if R !== nothing
                require_one_based_indexing(R)
                chkstride1(R)
                size(R, 1) >= n || throw(DimensionMismatch("R has too few rows: $(size(R,1)) < $n"))
                size(R, 2) >= n || throw(DimensionMismatch("R has too few columns: $(size(R,2)) < $n"))
            end
            
            lda = max(1, stride(V, 2))
            
            @no_escape begin
                # Allocate tau.
                tau_ptr = @alloc($elty, n)
                info_ptr = @alloc(BlasInt, 1)
                
                # geqrf!: first call queries optimal lwork.
                work_query = @alloc($elty, 1)
                lwork_query = BlasInt(-1)
                
                ccall((@blasfunc($geqrf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, pointer(V), lda,
                      pointer(tau_ptr), pointer(work_query), lwork_query, pointer(info_ptr))
                
                lwork_geqrf = round(BlasInt, nextfloat(real(unsafe_load(pointer(work_query)))))
                
                # geqrf!: actual factorization.
                work_geqrf = @alloc($elty, lwork_geqrf)
                
                ccall((@blasfunc($geqrf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, pointer(V), lda,
                      pointer(tau_ptr), pointer(work_geqrf), lwork_geqrf, pointer(info_ptr))
                
                chklapackerror(unsafe_load(pointer(info_ptr)))
                
                # Extract R into the output matrix and clear the lower triangle.
                if R !== nothing
                    @inbounds for j = 1:n
                        for i = 1:j
                            R[i, j] = V[i, j]
                        end
                        for i = j+1:n
                            R[i, j] = zero($elty)
                        end
                    end
                end
                
                # orgqr!/ungqr!: generate the orthogonal/unitary matrix.
                # Choose workspace and dimensions based on whether Q is provided.
                if Q !== nothing
                    @views copyto!(Q[:, 1:n], V[:, 1:n])
                end
                
                work_mat = Q === nothing ? V : Q
                lda_work = max(1, stride(work_mat, 2))
                nrows = Q === nothing ? m : m
                ncols = Q === nothing ? n : m
                
                # Query optimal lwork.
                lwork_query2 = BlasInt(-1)
                ccall((@blasfunc($xgqr), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      nrows, ncols, n, pointer(work_mat), lda_work,
                      pointer(tau_ptr), pointer(work_query), lwork_query2, pointer(info_ptr))
                
                lwork_xgqr = round(BlasInt, nextfloat(real(unsafe_load(pointer(work_query)))))
                work_xgqr = @alloc($elty, lwork_xgqr)
                
                # Actual QR reconstruction.
                ccall((@blasfunc($xgqr), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      nrows, ncols, n, pointer(work_mat), lda_work,
                      pointer(tau_ptr), pointer(work_xgqr), lwork_xgqr, pointer(info_ptr))
                
                chklapackerror(unsafe_load(pointer(info_ptr)))
                
                # If an external Q matrix was used, copy the thin Q back to V.
                if Q !== nothing
                    @views copyto!(V, Q[:, 1:n])
                end
                nothing
            end
            
            return nothing
        end
    end
end

function rls_qr!(
    V::AbstractMatrix{T},
    Q::Union{Nothing, AbstractMatrix{T}},
    R::Union{Nothing, AbstractMatrix{T}};
    type::Symbol = :lapack
) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}
    if type === :lapack
        return _rls_qr_lapack_impl!(V, Q, R)
    elseif type === :shiftchol
        Q === nothing || throw(ArgumentError("rls_qr!(; type = :shiftchol) only supports thin QR with Q === nothing"))
        return _rls_qr_shiftchol!(V, R)
    end
    throw(ArgumentError("unsupported rls_qr! type: $type; expected :lapack or :shiftchol"))
end


# for orthogonalization
@inline rls_qr!(V::AbstractMatrix{T}; type::Symbol = :lapack) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}} =
    rls_qr!(V, nothing, nothing; type = type)
# for thin QR factorization
@inline rls_qr!(V::AbstractMatrix{T}, R::AbstractMatrix{T}; type::Symbol = :lapack) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}} =
    rls_qr!(V, nothing, R; type = type)

@inline function _rls_set_identity!(A::AbstractMatrix{T}) where {T}
    m, n = size(A)
    m == n || throw(DimensionMismatch("identity initialization requires a square matrix, got $(size(A))"))
    fill!(A, zero(T))
    @inbounds for i = 1:n
        A[i, i] = one(T)
    end
    return nothing
end

@inline function _rls_add_diag_shift!(A::AbstractMatrix{T}, shift::Real) where {T}
    n1, n2 = size(A)
    n1 == n2 || throw(DimensionMismatch("diagonal shift requires a square matrix, got $(size(A))"))
    shiftT = convert(T, shift)
    @inbounds for i = 1:n1
        A[i, i] += shiftT
    end
    return nothing
end

@inline _rls_frobenius_norm(A::AbstractMatrix) = norm(A)

@inline function _rls_gnorm_sq(X::AbstractMatrix{T}) where {T}
    RT = typeof(abs2(zero(T)))
    max_col_norm_sq = zero(RT)
    @views @inbounds for j = 1:size(X, 2)
        col_norm = norm(X[:, j])
        max_col_norm_sq = max(max_col_norm_sq, col_norm * col_norm)
    end
    return max_col_norm_sq
end

for (elty, potrf, trtri, trmm) in ((:Float32, :spotrf_, :strtri_, :strmm_),
                                   (:Float64, :dpotrf_, :dtrtri_, :dtrmm_),
                                   (:ComplexF32, :cpotrf_, :ctrtri_, :ctrmm_),
                                   (:ComplexF64, :zpotrf_, :ztrtri_, :ztrmm_))
    @eval begin
        @inline function _rls_potrf_info!(uplo::AbstractChar, A::AbstractMatrix{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n = LinearAlgebra.checksquare(A)
            lda = max(1, stride(A, 2))
            lda == 0 && return 0

            info = zero(BlasInt)
            @no_escape begin
                info_ptr = @alloc(BlasInt, 1)
                ccall((@blasfunc($potrf), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
                      uplo, n, pointer(A), lda, pointer(info_ptr), 1)

                info = unsafe_load(pointer(info_ptr))
                info < 0 && throw(ArgumentError("LAPACK potrf! received an illegal argument at position $(-info)"))
            end
            return info
        end

        @inline function _rls_trtri!(uplo::AbstractChar, diag::AbstractChar, A::AbstractMatrix{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            n = LinearAlgebra.checksquare(A)
            lda = max(1, stride(A, 2))

            @no_escape begin
                info_ptr = @alloc(BlasInt, 1)
                ccall((@blasfunc($trtri), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Clong, Clong),
                      uplo, diag, n, pointer(A), lda, pointer(info_ptr), 1, 1)
                chklapackerror(unsafe_load(pointer(info_ptr)))
            end
            return nothing
        end

        @inline function _rls_trmm!(
            side::AbstractChar,
            uplo::AbstractChar,
            transa::AbstractChar,
            diag::AbstractChar,
            alpha::$elty,
            A::AbstractMatrix{$elty},
            B::AbstractMatrix{$elty}
        )
            require_one_based_indexing(A, B)
            chkstride1(A, B)
            m, n = size(B)
            nA = LinearAlgebra.checksquare(A)
            nA == (side == 'L' ? m : n) ||
                throw(DimensionMismatch("size of A, $(size(A)), does not match side=$side for B with size $(size(B))"))

            ccall((@blasfunc($trmm), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                   Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Clong, Clong, Clong, Clong),
                  side, uplo, transa, diag, m, n,
                  alpha, pointer(A), max(1, stride(A, 2)), pointer(B), max(1, stride(B, 2)),
                  1, 1, 1, 1)
            return nothing
        end
    end
end

"""
    _rls_qr_shiftchol!(X, R; itmax = 5) -> nothing

Internal shifted Cholesky-QR backend for `rls_qr!(; type = :shiftchol)`.

- `X` is overwritten by the orthonormalized basis.
- `R`, when provided, accumulates the total triangular factor on the left:
  `R = R_k * ... * R_1`.

The routine retries a failed Cholesky factorization by adding a diagonal shift
`11 * (m*n + n*(n+1)) * eps(real(T)) * ||X||_g^2` to the Gram matrix.
"""
function _rls_qr_shiftchol!(
    X::AbstractMatrix{T},
    R::Union{Nothing, AbstractMatrix{T}};
    itmax::Int = 3
) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}
    require_one_based_indexing(X)
    chkstride1(X)

    m, n = size(X)
    m >= n || throw(DimensionMismatch("shifted Cholesky-QR requires m >= n, got m=$m, n=$n"))
    itmax >= 1 || throw(ArgumentError("itmax must be >= 1, got $itmax"))

    if R !== nothing
        require_one_based_indexing(R)
        chkstride1(R)
        size(R, 1) >= n || throw(DimensionMismatch("R row dimension $(size(R, 1)) is smaller than n=$n"))
        size(R, 2) >= n || throw(DimensionMismatch("R column dimension $(size(R, 2)) is smaller than n=$n"))
    end

    n == 0 && return nothing

    epsT = eps(real(T))
    conv_tol = 2 * sqrt(float(n)) * epsT

    converged = false

    @no_escape begin
        work_R = @alloc(T, n, n)
        work_R_shift = @alloc(T, n, n)

        mul!(work_R, X', X)

        it = 1
        while it <= itmax
            copyto!(work_R_shift, work_R)
            info = _rls_potrf_info!('U', work_R_shift)

            if info == 0
                if it == 1
                    R === nothing || _rls_set_identity!(R)
                end

                if R !== nothing
                    _rls_trmm!('L', 'U', 'N', 'N', one(T), work_R_shift, R)
                end

                _rls_trtri!('U', 'N', work_R_shift)

                _rls_trmm!('R', 'U', 'N', 'N', one(T), work_R_shift, X)

                mul!(work_R, X', X)
                copyto!(work_R_shift, work_R)
                _rls_add_diag_shift!(work_R_shift, -one(real(T)))
                
                if _rls_frobenius_norm(work_R_shift) < conv_tol
                    converged = true
                    break
                end

                it += 1
            else
                shift = 11 * (m * n + n * (n + 1)) * epsT * _rls_gnorm_sq(X)
                _rls_add_diag_shift!(work_R, shift)
            end
        end
    end

    if !converged
        @warn "rls_qr!(type = :shiftchol) did not converge within $itmax iterations"
    end
    return nothing
end

#=
rls_eigen!(A, W, Z; jobz='V', range='A', uplo='U', vl=0, vu=0, il=1, iu=0, abstol=0) -> m

Allocation-free symmetric/Hermitian eigensolver wrapper using LAPACK syevr/heevr.
Writes eigenvalues to W and eigenvectors to Z, then returns the number of computed eigenvalues.

Arguments:
- A: input symmetric/Hermitian matrix, overwritten in place, n x n
- W: preallocated eigenvalue vector, length >= n
- Z: preallocated eigenvector matrix, n x n or larger
- jobz: 'V' computes eigenvectors, 'N' computes eigenvalues only
- range: 'A' all values, 'V' value interval [vl, vu], 'I' index interval [il, iu]
- uplo: 'U' uses the upper triangle, 'L' uses the lower triangle
- vl, vu, il, iu, abstol: same meaning as in LAPACK syevr/heevr

Returns:
- m: number of computed eigenvalues
=#

for (elty, syevr) in ((:Float32, :ssyevr_),
                      (:Float64, :dsyevr_))
    @eval begin
        function rls_eigen!(
            A::AbstractMatrix{$elty},
            W::AbstractVector{$elty},
            Z::AbstractMatrix{$elty};
            jobz::AbstractChar = 'V',
            range::AbstractChar = 'A',
            uplo::AbstractChar = 'U',
            vl::$elty = zero($elty),
            vu::$elty = zero($elty),
            il::Int = 1,
            iu::Int = 0,
            abstol::$elty = -one($elty)
        )
            require_one_based_indexing(A, W, Z)
            chkstride1(A, Z)

            n = size(A, 1)
            size(A, 2) == n || throw(DimensionMismatch("A must be square: got $(size(A))"))
            length(W) >= n || throw(DimensionMismatch("W is too short: $(length(W)) < $n"))
            size(Z, 1) >= n || throw(DimensionMismatch("Z has too few rows: $(size(Z,1)) < $n"))
            size(Z, 2) >= n || throw(DimensionMismatch("Z has too few columns: $(size(Z,2)) < $n"))

            lda = max(1, stride(A, 2))
            ldz = max(1, stride(Z, 2))

            @no_escape begin
                # isuppz needs length 2*max(1,m); since m <= n, 2n is enough.
                isuppz_ptr = @alloc(BlasInt, 2 * max(1, n))
                m_ptr = @alloc(BlasInt, 1)
                info_ptr = @alloc(BlasInt, 1)

                # workspace query
                work_query = @alloc($elty, 1)
                iwork_query = @alloc(BlasInt, 1)
                lwork_query = BlasInt(-1)
                liwork_query = BlasInt(-1)

                ccall((@blasfunc($syevr), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                       Ptr{BlasInt}),
                      jobz, range, uplo, n,
                      pointer(A), lda, vl, vu,
                      il, iu, abstol, pointer(m_ptr),
                      pointer(W), pointer(Z), ldz, pointer(isuppz_ptr),
                      pointer(work_query), lwork_query, pointer(iwork_query), liwork_query,
                      pointer(info_ptr))

                # Read the optimal workspace sizes.
                lwork = round(BlasInt, nextfloat(real(unsafe_load(pointer(work_query)))))
                liwork = unsafe_load(pointer(iwork_query))

                work_ptr = @alloc($elty, lwork)
                iwork_ptr = @alloc(BlasInt, liwork)

                # Actual eigensolver call.
                ccall((@blasfunc($syevr), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                       Ptr{BlasInt}),
                      jobz, range, uplo, n,
                      pointer(A), lda, vl, vu,
                      il, iu, abstol, pointer(m_ptr),
                      pointer(W), pointer(Z), ldz, pointer(isuppz_ptr),
                      pointer(work_ptr), lwork, pointer(iwork_ptr), liwork,
                      pointer(info_ptr))

                chklapackerror(unsafe_load(pointer(info_ptr)))
            end
            return unsafe_load(pointer(m_ptr))
        end
    end
end

for (elty, relty, heevr) in ((:ComplexF32, :Float32, :cheevr_),
                             (:ComplexF64, :Float64, :zheevr_))
    @eval begin
        function rls_eigen!(
            A::AbstractMatrix{$elty},
            W::AbstractVector{$relty},
            Z::AbstractMatrix{$elty};
            jobz::AbstractChar = 'V',
            range::AbstractChar = 'A',
            uplo::AbstractChar = 'U',
            vl::$relty = zero($relty),
            vu::$relty = zero($relty),
            il::Int = 1,
            iu::Int = 0,
            abstol::$relty = -one($relty)
        )
            require_one_based_indexing(A, W, Z)
            chkstride1(A, Z)

            n = size(A, 1)
            size(A, 2) == n || throw(DimensionMismatch("A must be square: got $(size(A))"))
            length(W) >= n || throw(DimensionMismatch("W is too short: $(length(W)) < $n"))
            size(Z, 1) >= n || throw(DimensionMismatch("Z has too few rows: $(size(Z,1)) < $n"))
            size(Z, 2) >= n || throw(DimensionMismatch("Z has too few columns: $(size(Z,2)) < $n"))

            lda = max(1, stride(A, 2))
            ldz = max(1, stride(Z, 2))

            @no_escape begin
                isuppz_ptr = @alloc(BlasInt, 2 * max(1, n))
                m_ptr = @alloc(BlasInt, 1)
                info_ptr = @alloc(BlasInt, 1)

                # workspace query
                work_query = @alloc($elty, 1)
                rwork_query = @alloc($relty, 1)
                iwork_query = @alloc(BlasInt, 1)
                lwork_query = BlasInt(-1)
                lrwork_query = BlasInt(-1)
                liwork_query = BlasInt(-1)

                ccall((@blasfunc($heevr), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ref{$relty}, Ref{$relty},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{$relty}, Ptr{BlasInt},
                       Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}),
                      jobz, range, uplo, n,
                      pointer(A), lda, vl, vu,
                      il, iu, abstol, pointer(m_ptr),
                      pointer(W), pointer(Z), ldz, pointer(isuppz_ptr),
                      pointer(work_query), lwork_query, pointer(rwork_query), lrwork_query,
                      pointer(iwork_query), liwork_query, pointer(info_ptr))

                lwork = round(BlasInt, nextfloat(real(unsafe_load(pointer(work_query)))))
                lrwork = round(BlasInt, nextfloat(unsafe_load(pointer(rwork_query))))
                liwork = unsafe_load(pointer(iwork_query))

                work_ptr = @alloc($elty, lwork)
                rwork_ptr = @alloc($relty, lrwork)
                iwork_ptr = @alloc(BlasInt, liwork)

                ccall((@blasfunc($heevr), libblastrampoline), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ref{$relty}, Ref{$relty},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{$relty}, Ptr{BlasInt},
                       Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}),
                      jobz, range, uplo, n,
                      pointer(A), lda, vl, vu,
                      il, iu, abstol, pointer(m_ptr),
                      pointer(W), pointer(Z), ldz, pointer(isuppz_ptr),
                      pointer(work_ptr), lwork, pointer(rwork_ptr), lrwork,
                      pointer(iwork_ptr), liwork, pointer(info_ptr))

                chklapackerror(unsafe_load(pointer(info_ptr)))
            end
            return unsafe_load(pointer(m_ptr))
        end
    end
end


#=
rls_lq!(V, Q, L) -> nothing

Allocation-free LQ factorization using Bumper workspace.
V is overwritten with the first m rows of Q, Q stores the full orthogonal/unitary matrix,
and L stores the lower-triangular factor.

Arguments:
- V: input matrix m x n, overwritten with the first m rows of Q; requires m <= n
- Q: preallocated orthogonal/unitary matrix output n x n, or nothing to form Q in V
- L: preallocated lower-triangular output m x m, or nothing to skip L extraction

Returns:
- nothing; results are stored in V, Q, and L in place
=#

# Generate LQ factorization methods; real types use orglq and complex types use unglq.
for (elty, gelqf, xglq) in ((:Float32,    :sgelqf_, :sorglq_),
                             (:Float64,    :dgelqf_, :dorglq_),
                             (:ComplexF32, :cgelqf_, :cunglq_),
                             (:ComplexF64, :zgelqf_, :zunglq_))
    @eval begin
        function rls_lq!(
            V::AbstractMatrix{$elty},
            Q::Union{Nothing, AbstractMatrix{$elty}},
            L::Union{Nothing, AbstractMatrix{$elty}}
        )
            require_one_based_indexing(V)
            chkstride1(V)
            
            m, n = size(V)
            @assert m <= n "LQ factorization requires m <= n, got m=$m, n=$n"

            # Check dimensions.
            if Q !== nothing
                require_one_based_indexing(Q)
                chkstride1(Q)
                size(Q, 1) >= n || throw(DimensionMismatch("Q has too few rows: $(size(Q,1)) < $n"))
                size(Q, 2) >= n || throw(DimensionMismatch("Q has too few columns: $(size(Q,2)) < $n"))
            end


            if L !== nothing
                require_one_based_indexing(L)
                chkstride1(L)
                size(L, 1) >= m || throw(DimensionMismatch("L has too few rows: $(size(L,1)) < $m"))
                size(L, 2) >= m || throw(DimensionMismatch("L has too few columns: $(size(L,2)) < $m"))
            end
            
            lda = max(1, stride(V, 2))
            
            @no_escape begin
                # Allocate tau.
                tau_ptr = @alloc($elty, m)
                info_ptr = @alloc(BlasInt, 1)
                
                # gelqf!: first call queries optimal lwork.
                work_query = @alloc($elty, 1)
                lwork_query = BlasInt(-1)
                
                ccall((@blasfunc($gelqf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, pointer(V), lda,
                      pointer(tau_ptr), pointer(work_query), lwork_query, pointer(info_ptr))
                
                lwork_gelqf = round(BlasInt, nextfloat(real(unsafe_load(pointer(work_query)))))
                
                # gelqf!: actual factorization.
                work_gelqf = @alloc($elty, lwork_gelqf)
                
                ccall((@blasfunc($gelqf), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, pointer(V), lda,
                      pointer(tau_ptr), pointer(work_gelqf), lwork_gelqf, pointer(info_ptr))
                
                chklapackerror(unsafe_load(pointer(info_ptr)))
                
                # Extract L into the output matrix and clear the upper triangle.
                if L !== nothing
                    @inbounds for j = 1:m
                        for i = 1:j-1
                            L[i, j] = zero($elty)
                        end
                        for i = j:m
                            L[i, j] = V[i, j]
                        end
                    end
                end
                
                # orglq!/unglq!: generate the orthogonal/unitary matrix.
                # Choose workspace and dimensions based on whether Q is provided.
                if Q !== nothing
                    @views copyto!(Q[1:m, :], V[1:m, :])
                end
                
                work_mat = Q === nothing ? V : Q
                lda_work = max(1, stride(work_mat, 2))
                nrows = Q === nothing ? m : n
                ncols = Q === nothing ? n : n
                
                # Query optimal lwork.
                lwork_query2 = BlasInt(-1)
                ccall((@blasfunc($xglq), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      nrows, ncols, m, pointer(work_mat), lda_work,
                      pointer(tau_ptr), pointer(work_query), lwork_query2, pointer(info_ptr))
                
                lwork_xglq = round(BlasInt, nextfloat(real(unsafe_load(pointer(work_query)))))
                work_xglq = @alloc($elty, lwork_xglq)
                
                # Actual LQ reconstruction.
                ccall((@blasfunc($xglq), libblastrampoline), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      nrows, ncols, m, pointer(work_mat), lda_work,
                      pointer(tau_ptr), pointer(work_xglq), lwork_xglq, pointer(info_ptr))
                
                chklapackerror(unsafe_load(pointer(info_ptr)))
                
                # If an external Q matrix was used, copy the thin Q back to V.
                if Q !== nothing
                    @views copyto!(V, Q[1:m, :])
                end
                nothing
            end
            
            return nothing
        end
    end
end


# for orthogonalization
@inline rls_lq!(V::AbstractMatrix{T}) where T = rls_lq!(V, nothing, nothing)
# for thin LQ factorization
@inline rls_lq!(V::AbstractMatrix{T}, L::AbstractMatrix{T}) where T = rls_lq!(V, nothing, L)


# Gram-Schmidt type orth
function rls_orth!(
    V::AbstractMatrix{T},
    X::AbstractMatrix{T},
    W::Union{Nothing, AbstractMatrix{T}},
    R::Union{Nothing, AbstractMatrix{T}};
    type::Symbol = :lapack
) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}
    nV = size(V, 2)
    nX = size(X, 2)
    @assert size(V, 1) == size(X, 1) "V and X must have the same number of rows"
    
    @no_escape begin
        W1 = @alloc(T, nV, nX)
        R1 = @alloc(T, nX, nX)
        mul!(W1, V', X)  # W = V' * X
        mul!(X, V, W1, -one(T), one(T))  # X = X - V * W
        rls_qr!(X, R1; type = type)
        
        # orth twice 
        W2 = @alloc(T, nV, nX)
        R2 = @alloc(T, nX, nX)
        mul!(W2, V', X)  # W = V' * X
        mul!(X, V, W2, -one(T), one(T))  # X = X - V * W
        rls_qr!(X, R2; type = type)

        if W !== nothing
            # W = W1 + W2 * R1
            mul!(W, W2, R1, one(T), zero(T))
            W .+= W1
        end

        if R !== nothing
            # R = R2 * R1
            mul!(R, R2, R1)
        end
        nothing
    end
end

rls_orth!(V::AbstractMatrix{T}, X::AbstractMatrix{T}; type::Symbol = :lapack) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}} =
    rls_orth!(V, X, nothing, nothing; type = type)

# Triple-pass Gram-Schmidt orthogonalization (for testing).
function rls_orth3!(
    V::AbstractMatrix{T},
    X::AbstractMatrix{T},
    W::Union{Nothing, AbstractMatrix{T}},
    R::Union{Nothing, AbstractMatrix{T}}
) where T
    nV = size(V, 2)
    nX = size(X, 2)
    @assert size(V, 1) == size(X, 1) "V and X must have the same number of rows"

    @no_escape begin
        W1 = @alloc(T, nV, nX)
        R1 = @alloc(T, nX, nX)
        mul!(W1, V', X)
        mul!(X, V, W1, -one(T), one(T))
        rls_qr!(X, R1)

        W2 = @alloc(T, nV, nX)
        R2 = @alloc(T, nX, nX)
        mul!(W2, V', X)
        mul!(X, V, W2, -one(T), one(T))
        rls_qr!(X, R2)

        W3 = @alloc(T, nV, nX)
        R3 = @alloc(T, nX, nX)
        mul!(W3, V', X)
        mul!(X, V, W3, -one(T), one(T))
        rls_qr!(X, R3)

        if W !== nothing || R !== nothing
            R21 = @alloc(T, nX, nX)
            mul!(R21, R2, R1)

            if W !== nothing
                # W = W1 + W2*R1 + W3*(R2*R1)
                mul!(W, W2, R1, one(T), zero(T))
                W .+= W1
                W3R = @alloc(T, nV, nX)
                mul!(W3R, W3, R21)
                W .+= W3R
            end

            if R !== nothing
                # R = R3*R2*R1
                mul!(R, R3, R21)
            end
        end
        nothing
    end
end

rls_orth3!(V::AbstractMatrix{T}, X::AbstractMatrix{T}) where T = rls_orth3!(V, X, nothing, nothing)

# Robust multi-pass Gram-Schmidt with adaptive stopping.
# Stop when norm(V' * X, Fro) <= tol after at least miniter passes.
function rls_orth_robust!(
    V::AbstractMatrix{T},
    X::AbstractMatrix{T},
    W::Union{Nothing, AbstractMatrix{T}},
    R::Union{Nothing, AbstractMatrix{T}};
    miniter::Int = 2,
    maxiter::Int = 4,
    tol::Real = eps(real(T)) * sqrt(float(size(V, 2))),
    type::Symbol = :lapack
) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}
    nV = size(V, 2)
    nX = size(X, 2)
    @assert size(V, 1) == size(X, 1) "V and X must have the same number of rows"
    miniter >= 1 || throw(ArgumentError("miniter must be >= 1, got $miniter"))
    maxiter >= miniter || throw(ArgumentError("maxiter must be >= miniter, got $maxiter < $miniter"))

    niter = 0
    @no_escape begin
        Wk = @alloc(T, nV, nX)
        Rk = @alloc(T, nX, nX)

        Wtot = W === nothing ? nothing : @alloc(T, nV, nX)
        Rtot = R === nothing ? nothing : @alloc(T, nX, nX)
        tmpW = W === nothing ? nothing : @alloc(T, nV, nX)
        tmpR = R === nothing ? nothing : @alloc(T, nX, nX)

        niter = 0
        for it = 1:maxiter
            mul!(Wk, V', X)
            mul!(X, V, Wk, -one(T), one(T))
            rls_qr!(X, Rk; type = type)

            if it == 1
                if Wtot !== nothing
                    copyto!(Wtot, Wk)
                end
                if Rtot !== nothing
                    copyto!(Rtot, Rk)
                end
            else
                if Wtot !== nothing
                    mul!(tmpW, Wk, Rtot)
                    Wtot .+= tmpW
                end
                if Rtot !== nothing
                    mul!(tmpR, Rk, Rtot)
                    copyto!(Rtot, tmpR)
                end
            end

            mul!(Wk, V', X)
            niter = it
            if it >= miniter && norm(Wk) <= tol
                break
            end
        end

        if W !== nothing
            copyto!(W, Wtot)
        end
        if R !== nothing
            copyto!(R, Rtot)
        end
        nothing
    end
    return niter
end

function rls_orth_robust!(
    V::AbstractMatrix{T},
    X::AbstractMatrix{T};
    miniter::Int = 2,
    maxiter::Int = 4,
    tol::Real = eps(real(T)) * sqrt(float(size(V, 2))),
    type::Symbol = :lapack
) where {T<:Union{Float32, Float64, ComplexF32, ComplexF64}}
    return rls_orth_robust!(V, X, nothing, nothing; miniter = miniter, maxiter = maxiter, tol = tol, type = type)
end
