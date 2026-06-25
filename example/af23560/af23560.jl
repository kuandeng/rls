include("../../src/RLS.jl")

C = loaddata("af23560")
C = ComplexF64.(C)
# p = qr(C).cpiv

normC = 645.74001457933
x_st, x_ed = -1.6, 0.8
y_st, y_ed = -1.55, 1.55
n_x, n_y = 200, 200
k = 20
rec = recursion_grid(x_st, x_ed, y_st, y_ed, n_x, n_y, k)