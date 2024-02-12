def is_symmetric(matrix):
    """In this function, it checks to see if the matrix is symmetric. It inputs the matrix, and
       since this is a boolean, if the function is symmetric, it returns true. If it's not symmetric,
       it outputs false."""
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

def is_positive_definite(matrix):
    """In this function, this one works much like the symmetric function. It inputs the matrix as a list,
       and if the matrix is definite positive, it will return true as a boolean. If it's false, it's not
       positive definite."""
    try:
        cholesky_decomposition(matrix)
        return True
    except ValueError:
        return False

def cholesky_decomposition(matrix):
    """In this function, it will operate if the matrix is symmetric and positive definite. The matrix is
       the input, and it will output a lower triangular matrix L such that the matrix = LL^T"""
    n = len(matrix)
    L = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i+1):
            if i == j:
                temp_sum = sum(L[i][k] ** 2 for k in range(j))
                L[i][j] = (matrix[i][i] - temp_sum) ** 0.5
            else:
                temp_sum = sum(L[i][k] * L[j][k] for k in range(j))
                L[i][j] = (matrix[i][j] - temp_sum) / L[j][j]

    return L

def forward_substitution(L, b):
    """In this function, it will perform a forward substitution to solve the system Ly=b. The 'L'
       is the lower triangular matrix, and the 'b' is the right-hand side vector. It will return the
       solution vector 'y'"""
    n = len(L)
    y = [0] * n

    for i in range(n):
        temp_sum = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - temp_sum) / L[i][i]

    return y

def backward_substitution(L, y):
    """In this function, it works pretty much exactly like the last function, but it is to solve the
        system LTx=y. 'L' is the lower triangular matrix, and 'y' is the vector obtained from forward
        substitution. It outputs the solution vector 'x'."""
    n = len(L)
    x = [0] * n

    for i in range(n-1, -1, -1):
        temp_sum = sum(L[j][i] * x[j] for j in range(i+1, n))
        x[i] = (y[i] - temp_sum) / L[i][i]

    return x

def doolittle_factorization(matrix):
    """In this function, it performs the LU factorization using the Doolittle method. The input is the
       matrix, and it returns a lower triangular matrix 'L' and an upper triangular matrix 'U' such that
       matrix = LU."""
    n = len(matrix)
    L = [[0] * n for _ in range(n)]
    U = [[0] * n for _ in range(n)]

    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            temp_sum = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = matrix[i][j] - temp_sum

        for j in range(i+1, n):
            temp_sum = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (matrix[j][i] - temp_sum) / U[i][i]

    return L, U

def doolittle_solve(L, U, b):
    """In this function, it solves the system using the Doolittle LU factorization. 'L' is the lower
       triangular matrix, 'U' is the upper triangular matrix, and 'b' is the right sie vector. It returns
       the solution vector 'x'."""
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def main():
    # Problem 1
    A1 = [[1, -1, 3, 2],
          [-1, 5, -5, -2],
          [3, -5, 19, 3],
          [2, -2, 3, 21]]

    b1 = [15, -35, 94, 1]

    if is_symmetric(A1) and is_positive_definite(A1):
        L1 = cholesky_decomposition(A1)
        y1 = forward_substitution(L1, b1)
        solution1 = backward_substitution(L1, y1)
        print("Cholesky method used:")
    else:
        L1, U1 = doolittle_factorization(A1)
        solution1 = doolittle_solve(L1, U1, b1)
        print("Doolittle method used:")

    print("Solution vector:", solution1)

    # Problem 2
    A2 = [[4, 2, 4, 0],
          [2, 2, 3, 2],
          [4, 3, 6, 3],
          [0, 2, 3, 9]]

    b2 = [20, 36, 60, 122]

    if is_symmetric(A2) and is_positive_definite(A2):
        L2 = cholesky_decomposition(A2)
        y2 = forward_substitution(L2, b2)
        solution2 = backward_substitution(L2, y2)
        print("\nCholesky method used:")
    else:
        L2, U2 = doolittle_factorization(A2)
        solution2 = doolittle_solve(L2, U2, b2)
        print("\nDoolittle method used:")

    print("Solution vector:", solution2)

if __name__ == "__main__":
    main()
