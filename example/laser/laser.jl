include("../../src/RLS.jl")

"""
    landau_demo(N, F)

Construct the discretized matrix B for the Landau laser integral equation.
This is a Julia version of EigTool's `landau_demo`.

- `N`: matrix dimension
- `F`: parameter; the original MATLAB code uses 32 when N > 200 and 12 otherwise

Returns:
- `B :: Matrix{ComplexF64}`
"""
function landau_demo(N::Integer, F::Real)
    N = Int(N)
    F = float(F)

    # Gaussian quadrature nodes and weights.
    # beta = 0.5*(1-(2*[1:N-1]).^(-2)).^(-0.5);
    k     = collect(1:N-1)
    beta  = 0.5 .* (1 .- (2.0 .* k).^(-2)) .^ (-0.5)

    # T = diag(beta,1) + diag(beta,-1);
    T = diagm(1 => beta, -1 => beta)

    # [V,D] = eig(T); nodes are the diagonal entries of D.
    # T is real symmetric, so eigen(Symmetric(T)) is sufficient.
    Fdec  = eigen(Symmetric(T))
    nodes = Fdec.values          # N real nodes.
    V     = Fdec.vectors         # Corresponding eigenvectors.

    # weights = 2*V(1,index).^2; eigen already returns sorted values here.
    weights = 2.0 .* (V[1, :].^2)    # 1 x N -> N-element vector.

    # Build matrix B.
    B = Matrix{ComplexF64}(undef, N, N)

    sqrt_iF = sqrt(im * F)       # sqrt(1i*F)
    for k in 1:N
        # MATLAB equivalent:
        # B(k,:) = weights(:)'*sqrt(1i*F).*exp(-1i*pi*F*(nodes(k)-nodes(:)').^2);
        # Keep row as a length-N vector (not 1×N matrix) to match B[k, :].
        row = sqrt_iF .* weights .* exp.(-im * π * F .* (nodes[k] .- nodes).^2)
        @views B[k, :] .= row
    end

    # Similarity transform with Gaussian weights: diag(sqrt(w))*B*diag(1./sqrt(w)).
    w = sqrt.(weights)
    for j in 1:N
        @views B[:, j] .= w .* B[:, j] ./ w[j]
    end

    return B
end

C = landau_demo(5000, 40*pi)

x_st, x_ed = -1.1, 1.2
y_st, y_ed = -1.1, 1.1
n_x, n_y = 200, 200
k = 10
rec = recursion_grid(x_st, x_ed, y_st, y_ed, n_x, n_y, k)
