import enum


class Detector(enum.Enum):
    LAR = 0
    TILE = 1
    TTEM = 2
    TTHEC = 3
    FCALEM = 5
    FCALHAD = 6


class CaloSampling(enum.Enum):
    PSB = 0
    PSE = 1
    EMB1 = 2
    EMB2 = 3
    EMB3 = 4
    TILECAL1 = 5
    TILECAL2 = 6
    TILECAL3 = 7
    TILEEXT1 = 8
    TILEEXT2 = 9
    TILEEXT3 = 10
    EMEC1 = 11
    EMEC2 = 12
    EMEC3 = 13
    HEC1 = 14
    HEC2 = 15
    HEC3 = 16
