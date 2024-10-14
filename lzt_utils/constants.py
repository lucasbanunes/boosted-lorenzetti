from collections import OrderedDict

N_RINGS = 100

RINGS_LAYERS = OrderedDict(
    PS=list(range(0, 8)),
    EM1=list(range(8, 72)),
    EM2=list(range(72, 80)),
    EM3=list(range(80, 88)),
    HAD1=list(range(88, 92)),
    HAD2=list(range(92, 96)),
    HAD3=list(range(96, 100)),
)

RINGS_LIMITS = [8 ,72, 80, 88, 92, 96]