import cv2
import numpy as np

from src.imgproc import ImgProc
from src.hough_circles import HoughCircles
from src.canny import Canny

if __name__ == '__main__':
    # --- Canny Edge Detection ---
    chessboard: np.ndarray = cv2.imread('data/chessboard.jpg')
    chessboardGrey: np.ndarray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
    chessboardGreySmoothed: np.ndarray = ImgProc.convolve2D(chessboardGrey, ImgProc.gaussianKernel(9, 3))

    chessboardEdges: np.ndarray = Canny.apply(chessboardGreySmoothed, lowThreshold=.3, highThreshold=.7)

    cv2.imwrite('results/chessboard.png', chessboardEdges)

    # --- HoughCircles ---
    circle: np.ndarray = cv2.imread('data/circle.png')
    circleGrey: np.ndarray = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)
    circleGreySmoothed: np.ndarray = ImgProc.convolve2D(circleGrey, ImgProc.gaussianKernel(9, 3))

    detectedCircles: list[HoughCircles.Circle] = HoughCircles.apply(circleGreySmoothed, 0.8, param1=.9, param2=30,
                                                                    minRadius=140,
                                                                    maxRadius=160)

    circleWithDetections = HoughCircles.plotCircles(circle, detectedCircles)

    cv2.imwrite('results/circle.png', circleWithDetections)
