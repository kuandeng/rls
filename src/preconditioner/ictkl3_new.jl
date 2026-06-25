function ictkl3_new!(
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

    info = 0
    st = 0

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

        jsize = min(lsize+saved, ind2)
        saved = saved + lsize - jsize

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


function ictkl3_new!(
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

    ictkl3_new!(
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
