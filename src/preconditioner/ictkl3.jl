
"""
    split2!(n::Int, ncut::Int, ix::Vector{Int}, x::AbstractVector)

Partition `ix` in place so that values referenced by `ix[ncut:n]`
are at least as large as those referenced by `ix[1:ncut-1]`.
`x` is read-only; `ix` is rearranged in place to match Fortran `split2`.
"""
function split2!(n::Int, ncut::Int, ix::AbstractVector{IT}, x::AbstractVector{T}) where {IT<:Integer, T<:FloatOrComplex}
    if n <= 1 || ncut <= 0 || ncut > n
        return
    end

    first = 1
    last  = n

    while first < last
        mid    = last
        midval = abs(x[Int(ix[mid])])

        # for j = last-1 down to first
        @inbounds for j = last-1:-1:first
            if abs(x[Int(ix[j])]) > midval
                mid -= 1
                # swap ix[mid] <-> ix[j]
                ix[mid], ix[j] = ix[j], ix[mid]
            end
        end

        # final swap: ix[mid] <-> ix[last]
        ix[mid], ix[last] = ix[last], ix[mid]

        # final test
        if mid > ncut
            last = mid - 1
        elseif mid < ncut
            first = mid + 1
        else
            break
        end
    end

    return
end

"""
    srtshai2!(n::Int, ix::Vector{Int})

Sort `ix[1:n]` in ascending order using the Fortran `srtshai2` Shell-sort scheme.
The stride starts at `ls = 2^floor(log2(2n)) - 1` and is halved each pass.
"""
function srtshai2!(n::Int, ix::AbstractVector{IT}) where {IT<:Integer}
    if n == 1
        return
    end

    # Matches mylog2(2*n, ii): ii = floor(log2(2n)).
    ii = floor(Int, log2(2n))
    ls = (1 << ii) - 1
    nhalf = n ÷ 2

    @inbounds for _lt = 1:ii
        if ls > nhalf
            ls ÷= 2
        else
            lls = n - ls
            for i = 1:lls
                is = i + ls
                la = ix[is]
                j  = i
                js = is
                # 100: gapped insertion with stride ls.
                while true
                    if la >= ix[j]
                        ix[js] = la
                        break
                    else
                        ix[js] = ix[j]
                        js = j
                        j  -= ls
                    end
                    if j < 1
                        ix[js] = la
                        break
                    end
                end
            end
            ls ÷= 2
        end
    end

    return
end

"""
    uxvsr4!(k::Int, arrayi::AbstractVector{<:Integer}, arrayr::AbstractVector)

Sort `arrayi[1:k]` in ascending order with Shell sort, and apply the
same permutation to `arrayr[1:k]`. Equivalent to Fortran `uxvsr4`.

- `arrayi`: integer vector to sort; views are accepted.
- `arrayr`: real or complex vector permuted with `arrayi`; views are accepted.
- Only the first k entries are reordered (1-based indexing).

The sort key is the value of `arrayi`; `arrayr` is only swapped alongside it.
"""
function uxvsr4!(k::Integer, arrayi::AbstractVector{<:Integer}, arrayr::AbstractVector{T}) where {T}
    k_int = Int(k)
    if k_int <= 1
        return
    end
    @assert 1 <= k_int <= length(arrayi) <= typemax(Int) "invalid k/arrayi length"
    @assert k_int <= length(arrayr) "arrayr is too short"


    # Shell-sort initialization matching the Fortran routine.
    khalf = k_int ÷ 2
    ii = floor(Int, log2(2k_int))
    ls = (1 << ii) - 1

    # -- shellsort loop
    @inbounds for _lt = 1:ii
        if ls > khalf
            ls ÷= 2
        else
            lls = k_int - ls
            for i = 1:lls
                is = i + ls
                la = arrayi[is]
                lb = arrayr[is]
                j  = i
                js = is
                # gapped insertion with gap = ls
                while true
                    if la >= arrayi[j]
                        arrayi[js] = la
                        arrayr[js] = lb
                        break
                    else
                        arrayi[js] = arrayi[j]
                        arrayr[js] = arrayr[j]
                        js = j
                        j -= ls
                    end
                    if j < 1
                        arrayi[js] = la
                        arrayr[js] = lb
                        break
                    end
                end
            end
            ls ÷= 2
        end
    end

    return
end


function ictkl3!(
    n::Int,
    ia::AbstractVector{IT}, ja::AbstractVector{IT}, aa::AbstractVector{T},
    lsize::Int,
    rsize::Int,
    keep::mi35_keep{T, RT},
    control::mi35_control{RT},
    d::AbstractVector{RT},
    il::AbstractVector{IT}, jl::AbstractVector{IT}, al::AbstractVector{T},
    startl::AbstractVector{IT}, listl::AbstractVector{IT},
    ir::AbstractVector{IT}, jr::AbstractVector{IT}, ar::AbstractVector{T},
    startr::AbstractVector{IT}, listr::AbstractVector{IT},
    wr01::AbstractVector{T}, wn02::AbstractVector{IT}
) where {IT<:Integer, T<:FloatOrComplex, RT<:AbstractFloat}

    # unpack control parameters 
    tau1  = control.tau1
    
    tau2  = control.tau2
    small = control.small
    iscale = control.iscale
    scale = keep.scale  
    jm = control.rrt ? 0 : 2  # jm=0 when rrt=true; jm=2 otherwise.

    # # work space
    # wr01 = fill(zero(T), n)
    # wn02 = Vector{Int}(undef, n)


    info = 0;

    # initialize L data structures
    startl .= 0
    listl .= 0
    saved = 0
    al[1] = aa[1]
    a_jstrt = ia[1]
    il[1] = 1

    # initialize R data structures
    savedr = 0
    ir[1] = 1
    listr .= 0 
    startr .= 0
    
    # main loop over columns
    for j = 1:n
        dj = d[j]
        # test for breakdown
        if dj < small
            info = -j
            println("Breakdown in column $j: dj = $dj")
            return info
        end

        ind2 = 0
        a_jstop = ia[j+1]-1    

        # Scatter the current A column, excluding the diagonal, into wr.
        @inbounds for k = a_jstrt+1:a_jstop
            i = ja[k]
            sca = iscale > 0 ? scale[i] * scale[j] : one(RT)
            wr01[i] = aa[k] * sca
            startl[i] = 1
            ind2 += 1
            wn02[ind2] = i
        end


        # Gather updates from previously completed columns through the L chain.
        lcurr = listl[j]
        while lcurr != 0
            lnext = listl[lcurr]
            kstrt = startl[lcurr]  
            temp  = al[kstrt]
            istrt = kstrt + 1
            istop = il[lcurr+1] - 1
           
            # update listl for the ja lnextrow
            if istrt <= istop
                startl[lcurr] = istrt
                lnextrow = jl[istrt]
                listl[lcurr] = listl[lnextrow]
                listl[lnextrow] = lcurr
            end

            # gather updates to L from previous L columns
            @inbounds for ii = istrt:istop
                k = jl[ii]
                t = -(al[ii] * temp')
                if startl[k] != 0
                    wr01[k] += t
                else
                    ind2 += 1
                    wn02[ind2] = k
                    startl[k] = 1
                    wr01[k] = t
                end
            end

            # gather updates to R from previous L columns
            rstrt = startr[lcurr]
            rstop = ir[lcurr+1] - 1
            @inbounds for ii = rstrt:rstop
                k = jr[ii]
                if k > j
                    t = -(ar[ii] * temp')  # for complex
                    if startl[k] != 0
                        wr01[k] += t
                    else
                        ind2 += 1
                        wn02[ind2] = k
                        startl[k] = 1
                        wr01[k] = t
                    end
                end
            end

            #back to next column having a nonzero in a ja i
            lcurr = lnext
        end

        rcurr = listr[j]
        while rcurr != 0
            rnext = listr[rcurr]
            kstrt = startr[rcurr]
            temp  = ar[kstrt]
            rstrt = kstrt + 1
            rstop = ir[rcurr+1] - 1

            # update listr for the ja rnextrow
            if rstrt <= rstop
                startr[rcurr] = rstrt
                rnextrow = jr[rstrt]
                listr[rcurr] = listr[rnextrow]
                listr[rnextrow] = rcurr
            end

            # gather L updates from previous R columns
            istrt = startl[rcurr]
            istop = il[rcurr+1] - 1
            @inbounds for ii = istrt:istop
                k = jl[ii]
                if k > j
                    t = -(al[ii] * temp')
                    if startl[k] != 0
                        wr01[k] += t
                    else
                        ind2 += 1
                        wn02[ind2] = k
                        startl[k] = 1
                        wr01[k] = t
                    end
                end
            end
            if jm == 0 
                @inbounds for ii = kstrt:rstop
                    k = jr[ii]
                    t = - ar[ii] * temp'   # for complex  
                    if k > j
                        if startl[k] != 0
                            wr01[k] = wr01[k] + t
                        end
                    elseif k == j
                        dj = dj + real(t)
                    end
                end
            end
            # back to next column of R having a nonzero in a ja i
            rcurr = rnext
        end

        if jm == 0 && dj < small
            println("Breakdown in column $j: dj = $dj")
            info = -j
            return info
        end

        djtemp = one(T)/sqrt(dj)    

        jj = il[j]
        jl[jj] = j
        al[jj] = djtemp

        @inbounds for ii = 1:ind2
            k = wn02[ii]
            wr01[k] = wr01[k] * djtemp
            startl[k] = 0
        end

        # jsize is number of entries to be included in L.
        asize = a_jstop - a_jstrt
        jsize = min(asize + lsize + saved, ind2)
        saved = saved + asize + lsize - jsize
        largep = ind2 - jsize + 1

        lind = il[j] + 1
        rind = ir[j]

        if ind2 >= 1
            # select jsize large entries at the end of wn02
            split2!(ind2, largep, wn02, wr01)

            # store accepted entries
            srtshai2!(jsize, view(wn02, largep:ind2)) 
            
            indr = largep
            @inbounds for i = largep:ind2
                k    = wn02[i]
                temp = wr01[k]
                if abs(temp) >= tau1
                    # entry is large enough to keep in L
                    d[k] -= RT(abs2(temp))
                    if d[k] <= small && k > j
                        println("Breakdown in column $j: dk = $(d[k])")
                        return -j
                    end
                    al[lind] = temp
                    jl[lind] = k
                    lind += 1
                elseif abs(temp) > RT(0) && abs(temp) > tau2
                    # entry is small but not too tiny for R
                    wn02[indr] = k
                    indr += 1
                end           
            end

            # sort R entries by values
            radd = min(indr - 1, rsize + savedr)
            larger = (indr - 1) - radd + 1

            split2!(indr - 1, larger, wn02, wr01) 

            savedr = savedr + rsize - radd

            if tau2 == zero(RT)
                @inbounds for ii = larger:indr-1
                    k    = wn02[ii]
                    temp = wr01[k]
                    wr01[k] = zero(T)
                    ar[rind] = temp
                    jr[rind] = k
                    rind += 1
                end
            else
                @inbounds for ii = larger:indr-1
                    k    = wn02[ii]
                    temp = wr01[k]
                    wr01[k] = zero(T)
                    # bug fix october 2022: need to check entries of R are not tiny
                    if abs(temp) > RT(tau2)
                        ar[rind] = temp
                        jr[rind] = k
                        rind += 1
                    end
                end
            end

            # -- throw away the smallest entries
            @inbounds for ii = 1:larger-1
                k = wn02[ii]
                wr01[k] = zero(T)
            end
        end

        uxvsr4!(rind - ir[j], view(jr, ir[j]:(rind-1)), view(ar, ir[j]:(rind-1)))

        if j < n
            # set a_jstrt, il and jl for next column
            a_jstrt   = a_jstop + 1
            il[j+1] = lind
            ir[j+1] = rind

            # define startl, startr and linked list entries (remember no diagonal entry for R)
            startl[j] = il[j] + 1
            if lind > il[j] + 1
                lnext = jl[il[j] + 1]
                listl[j] = listl[lnext]
                listl[lnext] = j
            end

            if rind > ir[j]
                startr[j] = ir[j]
                rnext = jr[ir[j]]
                listr[j] = listr[rnext]
                listr[rnext] = j
            else
                startr[j] = ir[j] + 1
            end
        else
            il[n+1] = lind
            ir[n+1] = rind   # added 5/10/22
        end

    end
end

function ictkl3!(
    n::Int,
    ia::AbstractVector{IT}, ja::AbstractVector{IT}, aa::AbstractVector{T},
    lsize::Int,
    rsize::Int,
    keep::mi35_keep{T, RT},
    control::mi35_control{RT},
    d::AbstractVector{RT},
    il::AbstractVector{IT}, jl::AbstractVector{IT}, al::AbstractVector{T},
    startl::AbstractVector{IT}, listl::AbstractVector{IT},
    ir::AbstractVector{IT}, jr::AbstractVector{IT}, ar::AbstractVector{T},
    startr::AbstractVector{IT}, listr::AbstractVector{IT}
) where {IT<:Integer, T<:FloatOrComplex, RT<:AbstractFloat}
    # work space
    wr01 = fill(zero(T), n)
    wn02 = Vector{IT}(undef, n)

    ictkl3!(
        n,
        ia, ja, aa,
        lsize,
        rsize,
        keep,
        control,
        d,
        il, jl, al,
        startl, listl,
        ir, jr, ar,
        startr, listr,
        wr01, wn02
    )
end
