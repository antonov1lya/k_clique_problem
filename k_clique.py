import graphblas as gb
from numba import njit, prange
import numpy as np
from math import factorial as f


@njit(parallel=True)
def get_index(I_S, I_B, J_B, I, J, size):
    for i in prange(size):
        I[i] = I_B[I_S[i]]
        J[i] = J_B[I_S[i]]


def get_I_J_K(B, M_ll):
    B = B.to_coo(values=False)
    S = (M_ll[B[0], :] & M_ll.T[B[1], :]).to_coo(values=False)

    K = S[1]

    size = S[0].shape[0]
    I = np.empty(shape=size, dtype='int64')
    J = np.empty(shape=size, dtype='int64')
    get_index(S[0], B[0], B[1], I, J, size)

    return I, J, K


def get_C_1(n):
    indexes = [i for i in range(n)]
    values = [True for i in range(n)]
    C_1 = gb.Matrix.from_coo(indexes, indexes, values, dtype='BOOL')
    return C_1


def get_C_2(A):
    C_1 = get_C_1(A.nrows)
    indexes = gb.select.triu(A).to_coo(values=False)
    C_2 = (C_1[indexes[0], :] | C_1[indexes[1], :]).new(dtype='BOOL')
    return C_2


def get_C_3(A):
    C_1 = get_C_1(A.nrows)
    A = A.dup(dtype='BOOL')

    B = gb.semiring.lor_land(A @ A).new(dtype='BOOL', mask=A.S)
    A << gb.select.triu(A)
    B << gb.select.triu(B)

    I, J, K = get_I_J_K(B, A)

    C_3 = (C_1[I, :] | C_1[J, :] | C_1[K, :]).new(dtype='BOOL')

    return C_3


def get_reduced_M_l_r(A, C_l, C_r, l, r):
    M_l_r = None

    if r == 1:
        M_l_r = gb.select.valueeq(
            gb.semiring.plus_times(C_l @ A), l
        ).new(dtype='BOOL')
    else:
        chunk = gb.select.valueeq(gb.semiring.plus_times(C_l @ A), l)
        M_l_r = gb.select.valueeq(
            gb.semiring.plus_times(chunk @ C_r.T), l * r
        ).new(dtype='BOOL')

    v = gb.Matrix.from_dense([[i for i in range(A.nrows)]])

    W_max = gb.semiring.max_times(C_l @ v.T)
    W_min = None
    if r == 1:
        W_min = v.T
    else:
        W_min = gb.semiring.min_times(C_r @ v.T)

    M_l_r(mask=M_l_r.S) << gb.semiring.any_lt(W_max @ W_min.T)
    M_l_r << gb.select.valueeq(M_l_r, True)

    return M_l_r


def increment_cliques(A, C_l, l):
    if l == 1:
        return get_C_2(A)
    if l == 2:
        return get_C_3(A)

    M_l1 = get_reduced_M_l_r(A, C_l, None, l, 1)
    I = M_l1.to_coo(values=False)

    C_1 = get_C_1(A.nrows)
    C_l1 = (C_l[I[0], :] | C_1[I[1], :]).new(dtype='BOOL')

    return C_l1


def matching_k_cliques(A, k):
    if k == 1:
        return get_C_1(A.nrows)
    if k == 2:
        return get_C_2(A)
    if k == 3:
        return get_C_3(A)

    l = k // 3
    r = k % 3

    C_3l = None
    if l == 1:
        C_3l = get_C_3(A)
    else:
        C_l = matching_k_cliques(A, l)
        M_l_l = get_reduced_M_l_r(A, C_l, C_l, l, l)
        B = gb.semiring.lor_land(M_l_l @ M_l_l).new(dtype='BOOL', mask=M_l_l.S)
        I, J, K = get_I_J_K(B, M_l_l)
        C_3l = (C_l[I, :] | C_l[J, :] | C_l[K, :]).new(dtype='BOOL')

    if r == 0:
        return C_3l

    C_3l_1 = increment_cliques(A, C_3l, 3*l)

    if r == 1:
        return C_3l_1

    C_3l_2 = increment_cliques(A, C_3l_1, 3*l+1)

    return C_3l_2


def get_M_l_r(A, C_l, C_r, l, r):
    if l != 1 and r == 1:
        M_l_r = gb.select.valueeq(
            gb.semiring.plus_times(C_l @ A), l
        ).new(dtype='BOOL')
        return M_l_r

    if l == 1 and r != 1:
        M_l_r = gb.select.valueeq(
            gb.semiring.plus_times(A @ C_r.T), r
        ).new(dtype='BOOL')
        return M_l_r

    chunk = gb.select.valueeq(gb.semiring.plus_times(C_l @ A), l)
    M_l_r = gb.select.valueeq(
        gb.semiring.plus_times(chunk @ C_r.T), l * r
    ).new(dtype='BOOL')
    return M_l_r


def counting_k_cliques(A, k):
    if k == 1:
        return A.nrows
    if k == 2:
        return A.reduce_scalar().value // 2

    l = k // 3
    r = k % 3

    if r == 0:
        M_l_l = None
        if l == 1:
            M_l_l = A
        elif l == 2:
            C_2 = get_C_2(A)
            M_l_l = get_M_l_r(A, C_2, C_2, 2, 2).dup(dtype='UINT64')
        else:
            C_l = matching_k_cliques(A, l)
            M_l_l = get_reduced_M_l_r(A, C_l, C_l, l, l).dup(dtype='UINT64')

        B = gb.semiring.plus_times(
            M_l_l @ M_l_l
        ).new(mask=M_l_l.S, dtype='UINT64')

        sum = B.reduce_scalar().value

        if l == 1 or l == 2:
            coef = f(3*l) // (f(l) * f(l) * f(l))
            return sum // coef
        else:
            return sum

    if r == 1:
        M_l_l = None
        M_l1_l = None
        if l == 1:
            C_2 = get_C_2(A)
            M_l_l = A
            M_l1_l = get_M_l_r(A, C_2, None, 2, 1).dup(dtype='UINT64')
        else:
            C_l = matching_k_cliques(A, l)
            C_l1 = increment_cliques(A, C_l, l)
            M_l_l = get_reduced_M_l_r(A, C_l, C_l, l, l).dup(dtype='UINT64')
            M_l1_l = get_reduced_M_l_r(
                A, C_l1, C_l, l+1, l
            ).dup(dtype='UINT64')

        B = gb.semiring.plus_times(
            M_l1_l @ M_l_l
        ).new(mask=M_l1_l.S, dtype='UINT64')

        sum = B.reduce_scalar().value

        if l == 1:
            coef = f(3*l+1) // (f(l+1) * f(l) * f(l))
            return sum // coef
        else:
            return sum

    if r == 2:
        M_l_l1 = None
        M_l1_l1 = None
        if l == 1:
            C_2 = get_C_2(A)
            M_l_l1 = get_M_l_r(A, None, C_2, 1, 2).dup(dtype='UINT64')
            M_l1_l1 = get_M_l_r(A, C_2, C_2, 2, 2).dup(dtype='UINT64')
        else:
            C_l = matching_k_cliques(A, l)
            C_l1 = increment_cliques(A, C_l, l)
            M_l_l1 = get_reduced_M_l_r(
                A, C_l, C_l1, l, l+1
            ).dup(dtype='UINT64')
            M_l1_l1 = get_reduced_M_l_r(
                A, C_l1, C_l1, l+1, l+1
            ).dup(dtype='UINT64')

        B = gb.semiring.plus_times(
            M_l_l1 @ M_l1_l1
        ).new(mask=M_l_l1.S, dtype='UINT64')

        sum = B.reduce_scalar().value

        if l == 1:
            coef = f(3*l+2) // (f(l+1) * f(l+1) * f(l))
            return sum // coef
        else:
            return sum
