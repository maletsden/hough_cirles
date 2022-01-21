from typing import Optional, NamedTuple, Union, Callable
from numba import njit
import numpy as np
import cv2
from .canny import Canny


class HoughCircles:
    class Accumulator(NamedTuple):
        accumulator: np.ndarray
        r: np.ndarray
        da: int
        db: int

    class Circle(NamedTuple):
        x: int
        y: int
        r: int
        votes: np.uint32

    @staticmethod
    def apply(image: np.ndarray, dp: float, param1: Optional[float] = 100., param2: Optional[np.uint32] = 100,
              minRadius: Optional[np.uint32] = 0, maxRadius: Optional[np.uint32] = 0) -> list[Circle]:
        """
        Applies the HoughTransform to the image to detect all the circles.

        :param image: 1-channel image (2-dimensional array)
        :param dp: the inverse ratio of the accumulator resolution to the image resolution.
        :param param1: the parameter for Canny edge detection (highThreshold, lowThreshold = param1 / 2)
        :param param2: the threshold for votes number
        :param minRadius: the min radius of the circle
        :param maxRadius: the max radius of the circle

        :return: the list of HoughCircles.Circle
        """

        accumulator: HoughCircles.Accumulator = HoughCircles.constructAccumulator(image, dp, minRadius, maxRadius)
        edgesAndGradients: Canny.EdgesAndGradients = Canny.applyWithGradients(image, param1 / 2, param1)
        HoughCircles.vote(edgesAndGradients, accumulator)
        HoughCircles.threshold(accumulator, param2)

        import cv2
        cv2.imshow("gradientMagnitude", edgesAndGradients.gradientMagnitude)
        cv2.waitKey(3000)

        return HoughCircles.accumulator2Circles(accumulator)

    @staticmethod
    def constructAccumulator(image: np.ndarray, dp: float, minRadius: Optional[np.uint32] = 0,
                             maxRadius: Optional[np.uint32] = 0) -> Accumulator:
        """
        Constructs the HoughCircles.Accumulator

        :param image: 1-channel image (2-dimensional array)
        :param dp:
        :param minRadius:
        :param maxRadius:

        :return: the HoughCircles.Accumulator
        """
        dp = max(dp, 0.)
        m: int = image.shape[0]
        n: int = image.shape[1]

        if not maxRadius:
            maxRadius = np.sqrt(m * m + n * n)

        aNum: int = int(m * dp)
        bNum: int = int(n * dp)
        da: int = int(np.ceil(m / aNum))
        db: int = int(np.ceil(n / bNum))

        r: np.ndarray = np.arange(minRadius, maxRadius)

        rNum: int = r.size

        accumulator: np.ndarray = np.zeros((aNum, bNum, rNum), dtype=np.uint32)

        return HoughCircles.Accumulator(accumulator, r, da, db)

    @staticmethod
    def vote(edgesAndGradients: Canny.EdgesAndGradients, accumulator: Accumulator) -> None:
        """
        Votes for the best circles parameters using Canny.STRONG edges.

        :param edgesAndGradients: detected Canny edges and correspondent gradients
        :param accumulator: the HoughCircles.Accumulator

        :return: None
        """
        HoughCircles.__voteImpl(edgesAndGradients.edges, edgesAndGradients.gradientDirection,
                                accumulator.accumulator, accumulator.r, accumulator.da, accumulator.db)

    @staticmethod
    @njit
    def __voteImpl(edges: np.ndarray, direction: np.ndarray, accumulator: np.ndarray, radius: np.ndarray, da: int,
                   db: int) -> None:
        """
        Votes for the best circles parameters using Canny.STRONG edges.

        :param edges: detected Canny edges
        :param direction: detected Sobel gradients
        :param accumulator: the HoughCircles.Accumulator
        :param radius: the array of all possible radius
        :param da: delta a parameter
        :param db: delta b parameter

        :return: None
        """
        m: int = edges.shape[0]
        n: int = edges.shape[1]

        isInImgSpace: Callable[[int, int], bool] = lambda x, y: (0 <= x < m) and (0 <= y < n)

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
        """
        Empty all the cells of the accumulator that have less votes than the threshold.

        :param accumulator: the HoughCircles.Accumulator
        :param threshold: the min votes threshold

        :return: None
        """
        accumulator.accumulator[accumulator.accumulator < threshold] = 0

    @staticmethod
    def accumulator2Circles(accumulator: Accumulator) -> list[Circle]:
        """
        Transforms the HoughCircles.Accumulator to the list of the HoughCircles.Circle

        :param accumulator: the HoughCircles.Accumulator

        :return: the list of HoughCircles.Circle
        """

        circles: list[HoughCircles.Circle] = []
        for a in range(accumulator.accumulator.shape[0]):
            x: int = accumulator.da * a
            for b in range(accumulator.accumulator.shape[1]):
                y: int = accumulator.db * b
                for k in range(accumulator.accumulator.shape[2]):
                    if not accumulator.accumulator[a, b, k]:
                        continue

                    r: int = accumulator.r[k]
                    votes: np.uint32 = accumulator.accumulator[a, b, k]
                    circles.append(HoughCircles.Circle(x, y, r, votes))

        return circles

    @staticmethod
    def plotCircles(image: np.ndarray, circles: list[Union[Circle, tuple[int, int, int]]],
                    circleColor: Optional[tuple[int, int, int]] = (0, 255, 0),
                    circleWidth: Optional[int] = 2) -> np.ndarray:
        """
        Plots the circles on the input image.

        :param image: 1-channel image (2-dimensional array)
        :param circles: the list of circles (list of tuples[x, y, r])
        :param circleColor: the desired color of the circle (tuple of 3 RGB values)
        :param circleWidth: the desired width of the circle

        :return: the copy of the input image with plotted circles (2-dimensional array)
        """
        result = image.copy()
        for circle in circles:
            result = cv2.circle(result, (circle.x, circle.y), circle.r, circleColor, circleWidth)
        return result
