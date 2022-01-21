from typing import Optional, NamedTuple
from numba import njit
import numpy as np
from .canny import Canny


class HoughCircles:
    class Accumulator(NamedTuple):
        accumulator: np.ndarray
        r: np.ndarray
        da: int
        db: int

    @staticmethod
    def apply(image: np.ndarray, dp: float, param1: Optional[float] = 100.,
              param2: Optional[np.uint32] = 100,
              minRadius: Optional[np.uint32] = 0,
              maxRadius: Optional[np.uint32] = 0) -> Accumulator:

        accumulator: HoughCircles.Accumulator = HoughCircles.constructAccumulator(image, dp, minRadius, maxRadius)
        edgesAndGradients: Canny.EdgesAndGradients = Canny.applyWithGradients(image, param1 / 2, param1)
        HoughCircles.vote(edgesAndGradients, accumulator)
        HoughCircles.threshold(accumulator, param2)

        return accumulator

    @staticmethod
    def constructAccumulator(image: np.ndarray, dp: float, minRadius: Optional[np.uint32] = 0,
                               maxRadius: Optional[np.uint32] = 0) -> Accumulator:
        dp = max(dp, 0.)
        m: int = image.shape[0]
        n: int = image.shape[1]

        if not maxRadius:
            maxRadius = np.sqrt(m * m + n * n)

        aNum: int = int(m * dp)
        bNum: int = int(n * dp)
        da: int = int(np.ceil(m / aNum))
        db: int = int(np.ceil(n / bNum))

        print(da, db)
        r: np.ndarray = np.arange(minRadius, maxRadius)

        rNum: int = r.size

        accumulator: np.ndarray = np.zeros((aNum, bNum, rNum), dtype=np.uint32)

        return HoughCircles.Accumulator(accumulator, r, da, db)

    @staticmethod
    def vote(edgesAndGradients: Canny.EdgesAndGradients, accumulator: Accumulator) -> None:
        return HoughCircles.__voteImpl(edgesAndGradients.edges, edgesAndGradients.gradientDirection,
                                       accumulator.accumulator, accumulator.r, accumulator.da, accumulator.db)

    @staticmethod
    @njit
    def __voteImpl(edges: np.ndarray, direction: np.ndarray, accumulator: np.ndarray, radius: np.ndarray, da: int,
                   db: int) -> None:
        m: int = edges.shape[0]
        n: int = edges.shape[1]

        isInImgSpace = lambda x, y: (0 <= x < m) and (0 <= y < n)

        # Utilize the gradient direction
        cosThetas: np.float32 = np.cos(direction)
        sinThetas: np.float32 = np.sin(direction)

        for x in range(m):
            for y in range(n):
                if not edges[x, y]:
                    continue

                for rIdx in range(radius.size):
                    r: int = radius[rIdx]
                    a: int = int(x - r * cosThetas[x, y])
                    b: int = int(y - r * sinThetas[x, y])

                    if not isInImgSpace(a, b):
                        continue

                    a = a // da
                    b = b // db

                    accumulator[a, b, rIdx] += 1

                    # Vote for neighbour bind (like smoothing in accumulator array)
                    if isInImgSpace(a - 1, b):
                        accumulator[a - 1, b, rIdx] += 1

                    if isInImgSpace(a, b - 1):
                        accumulator[a, b - 1, rIdx] += 1

                    if isInImgSpace(a + 1, b):
                        accumulator[a + 1, b, rIdx] += 1

                    if isInImgSpace(a, b + 1):
                        accumulator[a, b + 1, rIdx] += 1

                    if isInImgSpace(a - 1, b - 1):
                        accumulator[a - 1, b - 1, rIdx] += 1

                    if isInImgSpace(a + 1, b - 1):
                        accumulator[a + 1, b - 1, rIdx] += 1

                    if isInImgSpace(a + 1, b + 1):
                        accumulator[a + 1, b + 1, rIdx] += 1

                    if isInImgSpace(a - 1, b + 1):
                        accumulator[a - 1, b + 1, rIdx] += 1

    @staticmethod
    def threshold(accumulator: Accumulator, threshold: Optional[np.uint32] = 100) -> None:
        accumulator.accumulator[accumulator.accumulator < threshold] = 0
