
@inline function _fill_shifted_diag!(
    d::AbstractVector{RT},
    ptr,
    val,
    scale::AbstractVector{RT},
    control::mi35_control{RT},
    alpha::RT
) where {RT<:AbstractFloat}
    @inbounds for i in eachindex(d)
        di = val[ptr[i]]
        sca = control.iscale > 0 ? scale[i]^2 : one(RT)
        d[i] = real(di) * sca + alpha
    end
    return d
end

function _initial_mi35_shift(
    n::Int,
    ptr,
    val,
    scale::AbstractVector{RT},
    control::mi35_control{RT}
) where {RT<:AbstractFloat}
    shift = _normalize_mi35_shift_control(control)
    alpha = shift.alpha_in

    if alpha == zero(RT)
        @inbounds for i = 1:n
            di = val[ptr[i]]
            sca = control.iscale > 0 ? scale[i]^2 : one(RT)
            if norm(di) == zero(RT)
                alpha = shift.lowalpha
            elseif real(di) < zero(RT)
                alpha = max(alpha, -real(di) * sca)
            end
        end
        if alpha > zero(RT)
            alpha = max(alpha, shift.lowalpha)
        end
    end

    return (
        alpha = alpha,
        lowalpha = shift.lowalpha,
        shift_factor = shift.shift_factor,
        shift_factor2 = shift.shift_factor2
    )
end

function _run_ichsd_trial!(
    d,
    alpha,
    n,
    ptr,
    row,
    val,
    lsize,
    rsize,
    keep,
    control,
    ptrl,
    rowl,
    vall,
    startl,
    listl,
    ptrr,
    rowr,
    valr,
    startr,
    listr
)
    _fill_shifted_diag!(d, ptr, val, keep.scale, control, alpha)
    flag = ictkl3_new!(
        n,
        ptr,
        row,
        val,
        lsize,
        rsize,
        keep,
        control,
        d,
        ptrl,
        rowl,
        vall,
        startl,
        listl,
        ptrr,
        rowr,
        valr,
        startr,
        listr
    )
    return flag === nothing ? 0 : flag
end

function _save_successful_factorization(ptrl, rowl, vall, d, alpha)
    return (
        ptrl = copy(ptrl),
        rowl = copy(rowl),
        vall = copy(vall),
        d = copy(d),
        alpha = alpha
    )
end

function _restore_successful_factorization!(saved, ptrl, rowl, vall, d)
    copyto!(ptrl, saved.ptrl)
    copyto!(rowl, saved.rowl)
    copyto!(vall, saved.vall)
    copyto!(d, saved.d)
    return saved.alpha
end

function ichsd(
    n::Int,
    ptr::AbstractVector{IT}, row::AbstractVector{IT}, val::AbstractVector{T},
    lsize::Int, rsize::Int,
    keep::mi35_keep{T, RT, IK}, control::mi35_control{RT};
    out_index_type=Int
) where {T<:FloatOrComplex, RT<:AbstractFloat, IT<:Integer, IK<:Integer}

    (out_index_type isa Type && out_index_type <: Integer) || throw(ArgumentError("out_index_type must be <: Integer, got $out_index_type"))
    OT = out_index_type

    # allocate space for the preconditioner
    lsize = min(max(0, lsize), n-1)

    # compute L/R workspace sizes first (with overflow-safe arithmetic)
    nnzl_mul = try
        Base.Checked.checked_mul(lsize, n)
    catch
        throw(OverflowError("nnzl overflow in ichsd: lsize=$lsize, n=$n"))
    end
    nnzl = try
        Base.Checked.checked_add(n, nnzl_mul)
    catch
        throw(OverflowError("nnzl overflow in ichsd: n + lsize*n with n=$n, lsize=$lsize"))
    end
    # println("lsize = ", lsize, ", nnzl = ", nnzl)
    nnzr = try
        Base.Checked.checked_mul(max(0, rsize), n)
    catch
        throw(OverflowError("nnzr overflow in ichsd: rsize=$(max(0, rsize)), n=$n"))
    end

    max_it = typemax(IT)
    if n > max_it || (nnzl + 1) > max_it || (nnzr + 1) > max_it
        throw(ArgumentError(
            "internal index type=$IT overflow risk: n=$n, nnzl=$nnzl, nnzr=$nnzr, typemax($IT)=$max_it"
        ))
    end

    if OT !== Int
        max_ot = typemax(OT)
        if n > max_ot || (nnzl + 1) > max_ot || (nnzr + 1) > max_ot
            throw(ArgumentError(
                "out_index_type=$OT overflow risk: n=$n, nnzl=$nnzl, nnzr=$nnzr, typemax($OT)=$max_ot"
            ))
        end
    end

    # allocate space for L : diag + off diagonal
    startl = Vector{IT}(undef, n+1)
    listl  = Vector{IT}(undef, n+1)
    ptrr = Vector{IT}(undef, n+1)
    ptrl = Vector{IT}(undef, n+1)
    rowl = Vector{IT}(undef, nnzl)
    vall = Vector{T}(undef, nnzl)

    # max elements for R
    startr = Vector{IT}(undef, n+1)
    listr  = Vector{IT}(undef, n+1)
    rowr = Vector{IT}(undef, nnzr)
    valr = Vector{T}(undef, nnzr)

    scale = keep.scale
    d = Vector{RT}(undef, n)
    shift = _initial_mi35_shift(n, ptr, val, scale, control)
    alpha = shift.alpha
    lowalpha = shift.lowalpha
    shift_factor = shift.shift_factor
    shift_factor2 = shift.shift_factor2

    flag = 0
    alpha_old = zero(RT)
    nrestart = 0
    nshift = 0
    saved = nothing

    while true
        flag_previous = flag
        flag = _run_ichsd_trial!(
            d,
            alpha,
            n,
            ptr,
            row,
            val,
            lsize,
            rsize,
            keep,
            control,
            ptrl,
            rowl,
            vall,
            startl,
            listl,
            ptrr,
            rowr,
            valr,
            startr,
            listr
        )

        # Temporarily disable the MI35-style "successful shift backoff" path.
        # For now, once a factorization succeeds at the current shift, we accept it
        # directly instead of probing smaller shifts and restoring the last success.
        #
        # if flag == 0 && alpha == lowalpha
        #     if nrestart < control.maxshift
        #         alpha_old = alpha
        #         saved = _save_successful_factorization(ptrl, rowl, vall, d, alpha)
        #         lowalpha /= shift_factor2
        #         if nrestart > 0 && flag_previous == 0
        #             lowalpha /= shift_factor2
        #         end
        #         alpha = lowalpha
        #         nrestart += 1
        #         nshift += 1
        #         continue
        #     end
        # elseif flag < 0
        if flag < 0
            if saved !== nothing
                alpha = _restore_successful_factorization!(saved, ptrl, rowl, vall, d)
                flag = 0
                break
            end

            if alpha_old != zero(RT)
                alpha = alpha_old
            else
                alpha = max(shift_factor * alpha, lowalpha)
                if flag_previous != 0
                    close_breakdown = abs(flag - flag_previous) <= div(n, 10)
                    enough_headroom = n + flag >= div(n, 10)
                    if close_breakdown && enough_headroom
                        alpha = max(shift_factor * alpha, lowalpha)
                    end
                end
            end
            nshift += 1
            continue
        end

        break
    end

    return ptrl, rowl, vall, d
end

# Compatibility shim for old call sites that still pass `info`.
function ichsd(
    n::Int,
    ptr::AbstractVector{IT}, row::AbstractVector{IT}, val::AbstractVector{T},
    lsize::Int, rsize::Int,
    keep::mi35_keep{T, RT, IK}, control::mi35_control{RT}, _info;
    out_index_type=Int
) where {T<:FloatOrComplex, RT<:AbstractFloat, IT<:Integer, IK<:Integer}
    return ichsd(n, ptr, row, val, lsize, rsize, keep, control; out_index_type = out_index_type)
end
