import taichi as ti

@ti.func
def AxisDetection(left1, right1, left2, right2):
    leftbound = ti.max(left1, left2)
    rightbound = ti.min(right1, right2)

    return leftbound < rightbound
