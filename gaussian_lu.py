def gaussian_elimination(A, b):
    """Solve Ax = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]      # augmented matrix [A | b]

    # Forward elimination
    for k in range(n - 1):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))  # pivot row
        if abs(M[p][k]) < 1e-12:
            raise ValueError("Matrix is singular (or nearly singular)")
        M[k], M[p] = M[p], M[k]                           # row swap
        for i in range(k + 1, n):
            m = M[i][k] / M[k][k]                         # multiplier
            for j in range(k, n + 1):
                M[i][j] -= m * M[k][j]
    if abs(M[n - 1][n - 1]) < 1e-12:
        raise ValueError("Matrix is singular (or nearly singular)")

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x


def lu_decompose(A):
    """Return P (as a permutation list), L, U such that P*A = L*U."""
    n = len(A)
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    perm = list(range(n))
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(U[i][k]))
        if abs(U[p][k]) < 1e-12:
            raise ValueError("Matrix is singular (or nearly singular)")
        if p != k:
            U[k], U[p] = U[p], U[k]
            perm[k], perm[p] = perm[p], perm[k]
            for j in range(k):                            # swap already-built part of L
                L[k][j], L[p][j] = L[p][j], L[k][j]
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]                   # store multiplier in L
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]
    for i in range(n):
        L[i][i] = 1.0
    return perm, L, U


def lu_solve(perm, L, U, b):
    """Solve using a stored factorisation: Ly = Pb, then Ux = y."""
    n = len(L)
    pb = [b[perm[i]] for i in range(n)]
    y = [0.0] * n                                         # forward substitution
    for i in range(n):
        y[i] = pb[i] - sum(L[i][j] * y[j] for j in range(i))
    x = [0.0] * n                                         # back substitution
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x


if __name__ == "__main__":
    A = [[ 2.0,  1.0, -1.0],
         [-3.0, -1.0,  2.0],
         [-2.0,  1.0,  2.0]]
    b = [8.0, -11.0, -3.0]
    
    x = gaussian_elimination(A, b)
    print("Gaussian elimination:", [round(v, 6) for v in x])
    
    perm, L, U = lu_decompose(A)
    print("perm:", perm)
    for name, M in (("L", L), ("U", U)):
        print(name)
        for r in M: print("  ", [round(v, 4) for v in r])
    print("LU solve:", [round(v, 6) for v in lu_solve(perm, L, U, b)])
    b2 = [1.0, 0.0, 5.0]
    print("LU solve, new b:", [round(v, 6) for v in lu_solve(perm, L, U, b2)])
    print("Gauss, new b   :", [round(v, 6) for v in gaussian_elimination(A, b2)])
    
    # verification P*A = L*U
    n=3
    PA=[A[perm[i]] for i in range(n)]
    LU=[[sum(L[i][k]*U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    print("PA==LU:", all(abs(PA[i][j]-LU[i][j])<1e-9 for i in range(n) for j in range(n)))
    # residual
    res=[sum(A[i][j]*x[j] for j in range(n))-b[i] for i in range(n)]
    print("residual:", res)
    try:
        gaussian_elimination([[1.0,2.0],[2.0,4.0]],[3.0,6.0])
    except ValueError as e: print("Singular test:", e)
