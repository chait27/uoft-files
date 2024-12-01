import sympy as sp

# define symbols 
# x0, x1, x2, x3 = sp.symbols('x0 x1 x2 x3')
xp, xm, x2, x3 = sp.symbols('xp xm x2 x3')

# x^mu coordinate
# xup = sp.Matrix([x0, x1, x2, x3])
xup = sp.Matrix([xp, xm, x2, x3])

# W = sp.Function('W')(*xup)
W = sp.Function('W')(xp, x2, x3)

# downstairs metric g_\mu\nu
# gdndn = sp.Matrix(
#     [[1 + W/2, W/2, 0, 0],
#      [W/2, -1 + W/2, 0, 0],
#      [0, 0, -1, 0], 
#      [0, 0, 0, -1]]
# )
gdndn = sp.Matrix(
    [[W, 1, 0, 0],
     [1, 0, 0, 0],
     [0, 0, -1, 0], 
     [0, 0, 0, -1]]
)

# inverse upstairs metric g^\mu\nu
gupup = gdndn.inv(method="LU")
# gupup = sp.Matrix(
#     [[1 - W/2, W/2, 0, 0],
#      [W/2, -1 - W/2, 0, 0],
#      [0, 0, -1, 0], 
#      [0, 0, 0, -1]]
# )

# check if we get identity back
print("Check if g^{\mu\\nu} g_{\mu\\nu} = Identity")
print(sp.simplify(gupup * gdndn))
print("")

dim = 4

# empty 4x4x4 array for christoffels
Chrisupdndn = sp.MutableDenseNDimArray([0]*dim**3, (dim, dim, dim))

# loop over mu, nu, lambda and dummy index sigma
for mu in range(dim):
    for nu in range(dim):
        for lam in range(dim):
            inner_term = 0
            for sigma in range(dim):
                inner_term += gupup[mu, sigma] * (
                    sp.diff(gdndn[sigma, lam], xup[nu]) +
                    sp.diff(gdndn[sigma, nu], xup[lam]) -
                    sp.diff(gdndn[nu, lam], xup[sigma]))
            Chrisupdndn[mu, nu, lam] = sp.simplify(0.5 * inner_term)

print("Non-Zero entries of the Christoffels")
# non-zero christoffels
for mu in range(dim):
    for nu in range(dim):
        for lam in range(dim):
            if Chrisupdndn[mu, nu, lam] != 0:
                print(f"Gamma^{mu}_{nu}{lam} = {sp.latex(Chrisupdndn[mu, nu, lam])}")
print("")

# empty riemann tensor array R^\rho_{\sigma\mu\nu}
Riemupdndndn = sp.MutableDenseNDimArray([0]*dim**4, (dim, dim, dim, dim))

# compute the riemann tensor components using the christoffel symbols
for rho in range(dim):
    for sigma in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                term1 = sp.diff(Chrisupdndn[rho, sigma, nu], xup[mu])
                term2 = sp.diff(Chrisupdndn[rho, sigma, mu], xup[nu])
                term3 = 0
                term4 = 0
                for lam in range(3):
                    term3 += Chrisupdndn[lam, sigma, nu] * Chrisupdndn[rho, lam, mu]
                    term4 += Chrisupdndn[lam, sigma, mu] * Chrisupdndn[rho, lam, nu]
                Riemupdndndn[rho, sigma, mu, nu] = sp.simplify(term1 - term2 + term3 - term4)

print("Non-Zero entries of the Riemann tensor")
# print non-zero riemann tensor entries
for rho in range(dim):
    for sigma in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                if Riemupdndndn[rho, sigma, mu, nu] != 0:
                    print(f"R^{rho}_{sigma}{mu}{nu} = {sp.latex(Riemupdndndn[rho, sigma, mu, nu])}")
print("")

# compute the ricci tensor R_{\mu\nu} 
Riccdndn = sp.MutableDenseNDimArray([0]*dim**2, (dim, dim))
for mu in range(dim):
    for nu in range(dim):
        term = 0
        for lam in range(dim):
            term += Riemupdndndn[lam, mu, nu, lam]
        Riccdndn[mu, nu] = sp.simplify(term)

print("Non-Zero entries of the Ricci tensor")
# print non-zero ricci tensor components
for mu in range(dim):
    for nu in range(dim):
        if Riccdndn[mu, nu] != 0:
            print(f"R_{mu}{nu} = {Riccdndn[mu, nu]}")
print("")

# compute the ricci scalar R
RiccScal = 0
for mu in range(dim):
    for nu in range(dim):
        RiccScal += gupup[mu, nu] * Riccdndn[mu, nu]
print(f"Ricci Scalar is R = {RiccScal}")

# Solve for Lambda in terms of L
# Lambda_eq = {}
# for mu in range(3):
#     for nu in range(3):
#         Lambda_eq[(mu, nu)] = sp.simplify(Riccdndn[mu, nu] - 0.5 * gdndn[mu, nu] * RiccScal)

# # Solve for Lambda in terms of L
# Lambda = sp.symbols('Lambda')
# Lambda_value = sp.solve(Lambda_eq[2, 2] + Lambda * gdndn[2, 2], Lambda)
# print(f"Value of Lambda = {Lambda_value}")