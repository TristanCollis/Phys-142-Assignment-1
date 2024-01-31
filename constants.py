import numpy as np


PERIOD = 2 * np.pi
EPSILON = PERIOD / 128
TIME_STEPS = 128
T = np.linspace(0, PERIOD, TIME_STEPS)
ALPHA = 2
X_INITIAL = 3 / 4
X_MIN = -4
X_MAX = 4
PARTITIONS = 600
DX = (X_MAX - X_MIN) / PARTITIONS
X = np.linspace(X_MIN, X_MAX, PARTITIONS + 1)[..., np.newaxis]
