using LinearAlgebra
using MKLSparse
using SparseArrays
using LinearAlgebra.LAPACK:geqrf!, orgqr!, geqrt!
import LinearAlgebra: mul!, ldiv!, *, adjoint, size
import SparseArrays:adjoint, size
using MAT
using MatrixMarket
using Printf
using Parameters
using Random
using AMD
using DataStructures: Queue, enqueue!, dequeue!
using Serialization

if !isdefined(Main, :AbstractComplex)
    const AbstractComplex = Complex{T} where T <: AbstractFloat
end
if !isdefined(Main, :FloatOrComplex)
    const FloatOrComplex = Union{AbstractFloat, AbstractComplex}
end

include("./types/RLSMatrices.jl")
using .RLSMatrices

include("./types/RLSLinearAlgebra.jl")
include("./utils/loadmtx.jl")
include("./preconditioner/RLSPreconditioner.jl")

include("./core/recuGrid.jl")
include("./core/recySpace.jl")
include("./core/fastRaylRitz.jl")
include("./core/lobpcgSvd.jl")
include("./core/psTriRecy.jl")
include("./core/psPrecRecy.jl")
include("./baseline/psInvLanc.jl")
