# Symmetric permutation of a lower-triangular CSC matrix: A -> P' * A * P.
# Inputs: n, ptr, row, val, perm.
# Outputs are written to preallocated ptrp, rowp, valp; work is length n.
# Requires ptr[end]-1 == length(row) == length(val) == nnz. Columns need not be sorted.
function permute_lower_csc!(
    n::Int,
    ptr::AbstractVector{IT}, row::AbstractVector{IT}, val::AbstractVector{T},
    perm::AbstractVector{IT},
    ptrp::AbstractVector{IT}, rowp::AbstractVector{IT}, valp::AbstractVector{T},
    work::AbstractVector{IT},
) where {T, IT<:Integer}

    @assert length(ptr)  == n+1
    @assert length(perm) == n
    @assert length(work) == n
    nnz = Int(ptr[end] - one(IT))
    @assert length(row) == nnz && length(val) == nnz
    @assert length(rowp) == nnz && length(valp) == nnz && length(ptrp) == n+1

    # Count lower-triangular nonzeros per permuted column and reserve the first slot for the diagonal.
    @inbounds for j in 1:n
        work[j] = IT(0)
    end
    @inbounds for j in 1:n
        j2 = Int(perm[j])
        for k in Int(ptr[j]):Int(ptr[j+1]-one(IT))
            i  = Int(row[k])
            i2 = Int(perm[i])
            if i2 < j2
                work[i2] += one(IT)
            else
                work[j2] += one(IT)
            end
        end
    end

    # Prefix sum into column pointers, then reuse work as insertion cursors.
    @inbounds begin
        ptrp[1] = one(IT)
        for j in 1:n
            ptrp[j+1] = ptrp[j] + work[j]
            work[j] = ptrp[j] + one(IT)   # Cursor starts after the reserved diagonal slot.
        end
    end

    # Scatter entries with the same two-way logic as the original Fortran routine.
    @inbounds for j in 1:n
        j2 = Int(perm[j])
        for k in Int(ptr[j]):Int(ptr[j+1]-one(IT))
            i  = Int(row[k])
            i2 = Int(perm[i])
            if i2 == j2
                k1 = Int(ptrp[j2])   # The diagonal is fixed at the column head.
                rowp[k1] = IT(j2)
                valp[k1] = val[k]
            elseif i2 < j2
                k1 = Int(work[i2]); work[i2] += one(IT)
                rowp[k1] = IT(j2)     # Store in column i2, row j2 (> i2).
                valp[k1] = val[k]'
            else
                k1 = Int(work[j2]); work[j2] += one(IT)
                rowp[k1] = IT(i2)     # Store in column j2, row i2 (>= j2).
                valp[k1] = val[k]
            end
        end
    end

    return nothing
end

# Convenience wrapper that allocates output buffers.
function permute_lower_csc(
    n::Int, ptr::AbstractVector{IT}, row::AbstractVector{IT}, val::AbstractVector{T}, perm::AbstractVector{IT}
) where {T, IT<:Integer}
    nnz  = Int(ptr[end] - one(IT))
    ptrp = Vector{IT}(undef, n+1)
    rowp = Vector{IT}(undef, nnz)
    valp = Vector{T}(undef, nnz)
    work   = Vector{IT}(undef, n)
    permute_lower_csc!(n, ptr, row, val, perm, ptrp, rowp, valp, work)
    return ptrp, rowp, valp
end
