from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ._utils import draw3d_arrow


class ReferenceFrame:
    def __init__(
        self,
        origin: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        dz: np.ndarray,
        name: str,
        color: str = "tab:blue",
    ) -> None:
        self.origin = origin
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.name = name

        self.color = color

    def draw3d(
        self,
        head_length: float = 0.3,
        ax: Optional[Axes3D] = None,
    ) -> Axes3D:
        if ax is None:
            ax = plt.gca(projection="3d")

        ax.text(*self.origin + 0.5, f"({self.name})")
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dx,
            head_length=head_length,
            color="tab:red",
            name="x",
        )
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dy,
            head_length=head_length,
            color="tab:green",
            name="y",
        )
        ax = draw3d_arrow(
            ax=ax,
            arrow_location=self.origin,
            arrow_vector=self.dz,
            head_length=head_length,
            color="tab:blue",
            name="z",
        )
        return
