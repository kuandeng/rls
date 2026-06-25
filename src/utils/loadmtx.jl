function loaddata(mname::String; use_int32_index::Bool = true)
    mname*=".mtx"
    path = joinpath(@__DIR__, "..", "..", "data", mname)
    if !isfile(path)
        error("Matrix file not found at $path")
    end 
    A = mmread(path)

    # Normalize loaded matrices to ComplexF64.
    if A isa AbstractArray
        A = ComplexF64.(A)
    else
        A = convert.(ComplexF64, A)
    end

    # Use Int32 sparse indices when the matrix dimensions fit.
    if use_int32_index && A isa SparseMatrixCSC
        max_i32 = typemax(Int32)
        nrow, ncol = size(A)
        max_row = isempty(A.rowval) ? 0 : maximum(A.rowval)
        colptr_len = length(A.colptr)
        if nrow <= max_i32 && ncol <= max_i32 && max_row <= max_i32 && colptr_len <= max_i32
            A = SparseMatrixCSC{ComplexF64, Int32}(A)
        end
    end

    return A
end
