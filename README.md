# Hough Circle Transform

The Python implementation of Hough Transform to find circles in an image.

# Algorithm

 - Apply the Canny Edge Detection to find all the edges and correspondent gradients
 - Vote for the best parameters of the circles using Hough Transform algorithm
 - Threshold the accumulated votes to filter only the best fits

# Canny Edge Detection Example

| Input      | Output |
| ----------- | ----------- |
| ![Chessboard](https://github.com/maletsden/hough_cirles/blob/main/data/chessboard.jpg) | ![Chessboard_Edges](https://github.com/maletsden/hough_cirles/blob/main/results/chessboard.png) |

# Hough Circle Transform

| Input      | Output |
| ----------- | ----------- |
| ![Circle](https://github.com/maletsden/hough_cirles/blob/main/data/circle.png) | ![Chessboard_Edges](https://github.com/maletsden/hough_cirles/blob/main/results/circle.png) |
