import numpy as np
from numba import njit
from typing import Optional, Callable


class ImgProc:
    @staticmethod
    def gaussianKernel(size: int, sigma: Optional[float] = 1) -> np.ndarray:
        """
        Generates a 2D size x size Gaussian kernel

        :param size: the size of the kernel
        :param sigma: the standard deviation of the Gaussian distribution

        :return: the generated Gaussian kernel
        """

        if size <= 0:
            return np.array([])

        return ImgProc.__gaussianKernelImpl(size, sigma)

    @staticmethod
    @njit
    def __gaussianKernelImpl(size: int, sigma: Optional[float] = 1) -> np.ndarray:
        """
        Fills a 2D size x size Gaussian kernel

        :param size: the size of the kernel
        :param sigma: the standard deviation of the Gaussian distribution

        :return: the generated Gaussian kernel
        """

        kernel: np.ndarray = np.empty((size, size))

        k: int = int(size) // 2

        for i in range(size):
            for j in range(size):
                kernel[i, j] = -((i - k) * (i - k) + (j - k) * (j - k))

        twoSigmaSquared: float = 2.0 * sigma * sigma
        normal: float = 1.0 / (np.pi * twoSigmaSquared)
        kernel = normal * np.exp(kernel / twoSigmaSquared)

        return kernel

    @staticmethod
    @njit
    def crossCorrelation2D(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Cross-correlate two 2-dimensional arrays.

        Adds 0-padding around the image to provide the result of the same size as an input image.

        :param image: 1-channel image (2-dimensional array)
        :param kernel: (2k + 1)x(2k + 1) kernel

        :return: the results of cross-correlation (has the same size as an input image)
        """
        result: np.ndarray = np.empty_like(image)

        m: int = image.shape[0]
        n: int = image.shape[1]

        k1: int = kernel.shape[0] >> 1
        k2: int = kernel.shape[1] >> 1

        isInImgSpace: Callable[[int, int], bool] = lambda i, j: (0 <= i < m) and (0 <= j < n)

        for i in range(m):
            for j in range(n):
                value: float = 0.0
                for u in range(-k1, k1 + 1):
                    for v in range(-k2, k2 + 1):
                        if not isInImgSpace(i + u, j + v):
                            # padding with zeros
                            continue
                        value += kernel[u + k1, v + k2] * image[i + u, j + v]
                result[i, j] = value

        return result

    @staticmethod
    def convolve2D(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Convolve two 2-dimensional arrays.

        Adds 0-padding around the image to provide the result of the same size as an input image.

        :param image: 1-channel image (2-dimensional array)
        :param kernel: (2k + 1)x(2k + 1) kernel

        :return: result of the convolution (has the same size as an input image)
        """
        return ImgProc.crossCorrelation2D(image, np.flip(kernel))
