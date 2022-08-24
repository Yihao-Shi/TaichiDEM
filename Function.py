import taichi as ti


@ti.func
def Normalize(mv):
    mv_normalize = ti.Vector([0., 0., 0.])
    if not mv.norm() == 0:
        mv_normalize = mv.normalized()
    return mv_normalize


@ti.func
def sign(x):
    a = -1
    if x >= 0:
        a = 1
    return a


@ti.func
def SIGN(vector):
    for i in ti.static(range(vector.n)):
        if vector[i] > 0:
            vector[i] = vector[i]
        elif vector[i] < 0:
            vector[i] = -vector[i]
        elif vector[i] == 0:
            vector[i] = 0
    return vector


@ti.func
def Zero2One(x):
    k = 1
    if x == 1:
        k = 0
    return k


@ti.func
def Zero2OneVector(x):
    for i in ti.static(range(x.n)):
        if x[i] == 1:
            x[i] = 0
        else:
            x[i] = 1
    return x


@ti.func
def Max(i, j):
    m = j
    if i > j:
        m = i
    return m


@ti.func
def Min(i, j):
    m = j
    if i < j:
        m = i
    return m


@ti.func
def Diagonal(vec):
    return ti.Matrix([[vec[0], 0, 0], [0, vec[1], 0], [0, 0, vec[2]]])


def GreatestPowderOfTwo(length):
    k, q= 1, 0
    while k < length:
        k *= 2
        q += 1
    return int(k), int(q)


@ti.func
def EffectiveValue(x, y):
    return x * y / (x + y)


def NonNegative(x):
    if x <= 0:
        return 1
    else:
        return int(x)


@ti.func
def xor(a, b):
    return (a + b) & 1
