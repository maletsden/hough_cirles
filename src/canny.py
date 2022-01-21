import numpy as np
from numba import njit
from typing import Tuple, Final, Optional, NamedTuple
from .imgproc import ImgProc


class Canny:
    Edges = np.ndarray

    class EdgesAndGradients(NamedTuple):
        edges: np.ndarray
        gradientMagnitude: np.ndarray
        gradientDirection: np.ndarray

    WEAK: Final[np.uint8] = np.uint8(25)
    STRONG: Final[np.uint8] = np.uint8(255)

    Kx1: Final[np.ndarray] = np.array([[1], [2], [1]], np.float32)
    Kx2: Final[np.ndarray] = np.array([[-1, 0, 1]], np.float32)

    Ky1: Final[np.ndarray] = np.array([[1], [0], [-1]], np.float32)
    Ky2: Final[np.ndarray] = np.array([[1, 2, 1]], np.float32)

    @staticmethod
    def apply(image: np.ndarray, lowThreshold: Optional[float] = 0.05,
              highThreshold: Optional[float] = 0.15) -> Edges:
        """
        Applies the Canny edge detection algorithm to the input image.

        :param image: 1-channel image (2-dimensional array)
        :param lowThreshold: low threshold (the edges lower than the highThreshold and
                                            higher than lowThreshold will be marked as WEAK)
        :param highThreshold: high threshold (the edges higher than the highThreshold will be marked as STRONG)

        :return: detected STRONG edges
        """
        return Canny.applyWithGradients(image, lowThreshold, highThreshold).edges

    @staticmethod
    def applyWithGradients(image: np.ndarray, lowThreshold: Optional[float] = 0.05,
                           highThreshold: Optional[float] = 0.15) -> EdgesAndGradients:
        """
        Applies the Canny edge detection algorithm to the input image.

        :param image: 1-channel image (2-dimensional array)
        :param lowThreshold: low threshold (the edges lower than the highThreshold and
                                            higher than lowThreshold will be marked as WEAK)
        :param highThreshold: high threshold (the edges higher than the highThreshold will be marked as STRONG)

        :return: (detected STRONG edges, gradients magnitudes, gradients directions)
        """
        magnitude, direction = Canny.sobelFilters(image)
        suppressedEdges = Canny.nonMaxSuppression(magnitude, direction)
        thresholdEdges = Canny.threshold(suppressedEdges, lowThreshold, highThreshold)
        return Canny.EdgesAndGradients(edges=Canny.hysteresis(thresholdEdges), gradientMagnitude=magnitude,
                                       gradientDirection=direction)

    @staticmethod
    def sobelFilters(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies Sobel filter to the input image.

        Returns the magnitudes and the direction of the gradients in the each point.

        :param image: 1-channel image (2-dimensional array)

        :return: the magnitudes and the direction of the gradients in the each point
        """

        Ix: np.ndarray = ImgProc.convolve2D(image, Canny.Kx1)
        Ix: np.ndarray = ImgProc.convolve2D(Ix, Canny.Kx2)
        Iy: np.ndarray = ImgProc.convolve2D(image, Canny.Ky1)
        Iy: np.ndarray = ImgProc.convolve2D(Iy, Canny.Ky2)

        gradientMagnitude: np.ndarray = np.sqrt(np.square(Ix) + np.square(Iy))
        gradientMagnitude *= 255.0 / gradientMagnitude.max()

        gradientDirection: np.ndarray = np.arctan2(Iy, Ix)

        return gradientMagnitude.astype(np.uint8), gradientDirection.astype(np.float32)

    @staticmethod
    def nonMaxSuppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Suppress all the pixels (of the gradient intensity matrix) that is non maximum in the pixel's gradient direction.

        :param magnitude: gradient intensity matrix (the 2-dimensional array)
        :param direction: gradient direction matrix (the 2-dimensional array) (in radians)

        :return: the matrix of non-suppressed pixels
        """

        direction = direction * 180. / np.pi
        direction[direction < 0] += 180.

        return Canny.__nonMaxSuppressionImpl(magnitude, direction)

    @staticmethod
    @njit
    def __nonMaxSuppressionImpl(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Suppress all the pixels (of the gradient intensity matrix) that is non maximum in the pixel's gradient direction.

        :param magnitude: gradient intensity matrix (the 2-dimensional array)
        :param direction: gradient direction matrix (the 2-dimensional array)

        :return: the matrix of non-suppressed pixels
        """
        result: np.ndarray = np.empty_like(magnitude, dtype=np.uint8)

        m: int = magnitude.shape[0]
        n: int = magnitude.shape[1]

        top: np.uint8 = np.uint8(255)
        bottom: np.uint8 = np.uint8(255)

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                # angle 0
                if (0 <= direction[i, j] < (np.pi / 8)) or ((7 * np.pi / 8) <= direction[i, j] <= 180):
                    top = magnitude[i, j + 1]
                    bottom = magnitude[i, j - 1]
                # angle 45
                elif (np.pi / 8) <= direction[i, j] < (3 * np.pi / 8):
                    top = magnitude[i + 1, j - 1]
                    bottom = magnitude[i - 1, j + 1]
                # angle 90
                elif (3 * np.pi / 8) <= direction[i, j] < (5 * np.pi / 8):
                    top = magnitude[i + 1, j]
                    bottom = magnitude[i - 1, j]
                # angle 135
                elif (5 * np.pi / 8) <= direction[i, j] < (7 * np.pi / 8):
                    top = magnitude[i - 1, j - 1]
                    bottom = magnitude[i + 1, j + 1]

                if (magnitude[i, j] >= top) and (magnitude[i, j] >= bottom):
                    result[i, j] = magnitude[i, j]

        return result

    @staticmethod
    def threshold(edges: np.ndarray, lowThreshold: Optional[float] = 0.05,
                  highThreshold: Optional[float] = 0.15) -> np.ndarray:
        """
        Threshold detected edges.

        Marks all the edges that are higher than the highThreshold as STRONG.
        Marks all the edges that are lower than the highThreshold and higher than lowThreshold as WEAK.
        Deletes all other edges.

        :param edges: gradient intensity matrix (the 2-dimensional array)
        :param lowThreshold: low threshold (the edges lower than the highThreshold and
                                            higher than lowThreshold will be marked as WEAK)
        :param highThreshold: high threshold (the edges higher than the highThreshold will be marked as STRONG)

        :return: edges' marks (STRONG, WEAK and zeros).
        """
        return Canny.__thresholdImpl(edges, lowThreshold=lowThreshold, highThreshold=highThreshold,
                                     strong=Canny.STRONG, weak=Canny.WEAK)

    @staticmethod
    @njit
    def __thresholdImpl(edges: np.ndarray, lowThreshold: Optional[float] = 0.05, highThreshold: Optional[float] = 0.15,
                        strong: Optional[np.uint8] = 225, weak: Optional[np.uint8] = 112) -> np.ndarray:
        """
        Threshold detected edges.

        Marks all the edges that are higher than the highThreshold as STRONG.
        Marks all the edges that are lower than the highThreshold and higher than lowThreshold as WEAK.
        Deletes all other edges.

        :param edges: gradient intensity matrix (the 2-dimensional array)
        :param lowThreshold: low threshold
        :param highThreshold: high threshold
        :param strong: the value used to mark edge as STRONG
        :param weak: the value used to mark edge as WEAK

        :return: edges' marks (STRONG, WEAK and zeros).
        """

        maxVal: np.int8 = edges.max()
        highThreshold: float = maxVal * highThreshold
        lowThreshold: float = maxVal * lowThreshold

        m: int = edges.shape[0]
        n: int = edges.shape[1]

        mask: np.ndarray = np.zeros_like(edges, dtype=np.uint8)

        for i in range(m):
            for j in range(n):
                if edges[i, j] >= highThreshold:
                    mask[i, j] = strong
                elif lowThreshold <= edges[i, j]:
                    mask[i, j] = weak

        return mask

    @staticmethod
    def hysteresis(edges: np.ndarray) -> np.ndarray:
        """
        Hysteresis transforms WEAK pixels into STRONG ones,
        if and only if at least one of the pixels around the one being processed is a STRONG one.

        :param edges: gradient intensity matrix (the 2-dimensional array)

        :return: edges' marks (STRONG and zeros).
        """
        return Canny.__hysteresisImpl(edges, Canny.STRONG, Canny.WEAK)

    @staticmethod
    @njit
    def __hysteresisImpl(edges: np.ndarray, strong: np.uint8, weak: np.uint8) -> np.ndarray:
        """
        Hysteresis transforms WEAK pixels into STRONG ones,
        if and only if at least one of the pixels around the one being processed is a STRONG one.

        :param edges: gradient intensity matrix (the 2-dimensional array)
        :param strong: the value used to mark edge as STRONG
        :param weak: the value used to mark edge as WEAK

        :return: edges' marks (STRONG and zeros).
        """

        m: int = edges.shape[0]
        n: int = edges.shape[1]

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if edges[i, j] != weak:
                    continue

                value: np.uint8 = (edges[i + 1, j - 1] |
                                   edges[i + 1, j] |
                                   edges[i + 1, j + 1] |
                                   edges[i, j - 1] |
                                   edges[i, j + 1] |
                                   edges[i - 1, j - 1] |
                                   edges[i - 1, j] |
                                   edges[i - 1, j + 1])

                edges[i, j] = (strong if value == strong else 0)

        return edges
