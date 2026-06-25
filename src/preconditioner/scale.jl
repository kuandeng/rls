

"""
compute_scaling!(n, ia, ja, aa, iscale, scalesj, st, info)

- `iscale` controls the scaling mode:
    1: L2 scaling (Lin and More)
    2: mc77 scaling (skipped)
    3: mc64 scaling (skipped)
    4: diagonal scaling
- Fill `scalesj` with one scaling factor per node.
# - Returns status `st`; `info.flag` records error details.
"""
function compute_scaling!(
    n::Int,
    ia::AbstractVector{IT},
    ja::AbstractVector{IT},
    aa::AbstractVector{T},
    iscale::Int,
    scalesj::AbstractVector{T1}
) where {T, T1<:AbstractFloat, IT<:Integer}

    # Initialize status.
    st = 0

    if iscale == 1
        # 1: L2 scaling (Lin and More).
        scalesj .= zero(T1)  # Initialize.
        @inbounds for i in 1:n
            jstrt = ia[i]
            jstop = ia[i+1] - 1
            @inbounds for j in jstrt:jstop
                k = ja[j]
                temp = aa[j]
                temp2 = norm(temp)^2
                scalesj[k] += temp2
                if k == i
                    continue
                end
                scalesj[i] += temp2
            end
        end

        # Compute inverse L2 norms.
        @inbounds for i in 1:n
            scalesj[i] = sqrt(scalesj[i])
        end

        # Avoid division by zero.
        @inbounds for i in 1:n
            if scalesj[i] > zero(T1)
                scalesj[i] = one(T1) / sqrt(scalesj[i])
            else
                scalesj[i] = one(T1)
            end
        end

    elseif iscale == 2
        # 2: mc77 scaling - currently skipped.
        # TODO: Call mc77_scale when available.
        return
    elseif iscale == 3
        # 3: mc64 scaling - currently skipped.
        # TODO: Call mc64_scale when available.
        return

    elseif iscale == 4
        # 4: Diagonal scaling.
        @inbounds for i in 1:n
            jstrt = ia[i]
            scalesj[i] = aa[jstrt]
            if scalesj[i] > zero(T1)
                scalesj[i] = one(T1) / sqrt(scalesj[i])
            else
                scalesj[i] = one(T1)
            end
        end
    end

    return
end

function compute_scaling(
    n::Int,
    ia::AbstractVector{IT},
    ja::AbstractVector{IT},
    aa::AbstractVector{T},
    iscale::Int
) where {T, IT<:Integer}
    RT = real(T)
    scalesj = Vector{RT}(undef, n)
    compute_scaling!(n, ia, ja, aa, iscale, scalesj)
    return scalesj
end
