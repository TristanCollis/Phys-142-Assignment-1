from animate import animate_save
from constants import X

import numpy as np

from typing import Any


def display(pdf_vs_time: np.ndarray[float, Any]) -> None:
    animate_save(pdf_vs_time, X)
