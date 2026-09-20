import sympy as sp


def frobenius_norm_squared(matrix):
    return sp.simplify(sum(sp.conjugate(x) * x for x in matrix))


def check_3_by_3():
    i = sp.I
    a = sp.Matrix([
        [7 + i, 4 - i, 10],
        [-1, -i, -1 + i],
        [-6, -3 + i, -7],
    ])
    b = sp.Matrix([
        [4 - 4*i, 1 + i, 3 + 7*i],
        [4, i, -1 + 3*i],
        [10, 1 + i, -1 + 7*i],
    ])

    norm_a = frobenius_norm_squared(a)
    norm_b = frobenius_norm_squared(b)
    commutator_numerator = frobenius_norm_squared(2*a*b - b*a)
    ratio = sp.factor(commutator_numerator / (4 * norm_a * norm_b))

    assert sp.trace(a) == 0
    assert norm_a == 266
    assert norm_b == 271
    assert commutator_numerator == 360578
    assert ratio == sp.Rational(180289, 144172)
    assert ratio > sp.Rational(5, 4)

    print("3x3 example")
    print("  tr(A) =", sp.trace(a))
    print("  tr(B) =", sp.trace(b))
    print("  ||A||_F^2 =", norm_a)
    print("  ||B||_F^2 =", norm_b)
    print("  ||2AB-BA||_F^2 =", commutator_numerator)
    print("  R_(1/2) =", ratio, "> 5/4")


def check_5_by_5():
    i = sp.I
    a = sp.Matrix([
        [-17-3*i, 3+3*i, 48, -1-16*i, -14+38*i],
        [-1-3*i, 1-3*i, 3+7*i, 2-i, -6],
        [-8+3*i, 2+i, 19-14*i, -5-6*i, 1+17*i],
        [-14-6*i, 1+3*i, 37+15*i, 6-16*i, -23+24*i],
        [-18+4*i, 4+3*i, 45-6*i, -3-14*i, -9+36*i],
    ])
    b = sp.Matrix([
        [-17-13*i, -2+i, -2-18*i, -21-5*i, -7-10*i],
        [5-2*i, -4*i, 4+2*i, 4-4*i, 3],
        [45+9*i, 3-4*i, 11+37*i, 48, 21+14*i],
        [-3+12*i, 1+i, -12+2*i, 9*i, -3+4*i],
        [-6-44*i, -5-2*i, 32-23*i, -16-43*i, 6-29*i],
    ])

    norm_a = frobenius_norm_squared(a)
    norm_b = frobenius_norm_squared(b)
    commutator_numerator = frobenius_norm_squared(2*a*b - b*a)
    ratio = sp.factor(commutator_numerator / (4 * norm_a * norm_b))

    assert sp.trace(a) == 0
    assert sp.trace(b) == 0
    assert norm_a == 12863
    assert norm_b == 15019
    assert commutator_numerator == 966015585
    assert ratio == sp.Rational(966015585, 772757588)
    assert ratio > sp.Rational(5, 4)

    print("5x5 example")
    print("  tr(A) =", sp.trace(a))
    print("  tr(B) =", sp.trace(b))
    print("  ||A||_F^2 =", norm_a)
    print("  ||B||_F^2 =", norm_b)
    print("  ||2AB-BA||_F^2 =", commutator_numerator)
    print("  R_(1/2) =", ratio, "> 5/4")


if __name__ == "__main__":
    check_3_by_3()
    check_5_by_5()
    print("all exact checks passed")
