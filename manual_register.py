# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:36:48 2024

@author: janss
"""

import cv2
import math
import numpy as np

# Global variable to store clicked coordinates
clicked_coordinates = []


def on_click(event: int, x: int, y: int, flags, param) -> None:
    """
    Callback function for mouse click event.

    Args:
        event (int): The type of mouse event.
        x (int): The x-coordinate of the mouse click.
        y (int): The y-coordinate of the mouse click.
        flags (int): Additional flags passed by OpenCV.
        param (object): Additional parameters passed by OpenCV.
    """

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coordinates.append((x, y))


def crop_center(image: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
    """
    Crop the center portion of an image.

    Args:
        image (numpy.ndarray): The input image.
        crop_width (int): The width of the cropped region.
        crop_height (int): The height of the cropped region.

    Returns:
        numpy.ndarray: The cropped image.
    """
    image_height, image_width = image.shape[:2]
    start_x = max(image_width // 2 - (crop_width // 2), 0)
    start_y = max(image_height // 2 - (crop_height // 2), 0)
    end_x = min(start_x + crop_width, image_width)
    end_y = min(start_y + crop_height, image_height)
    return image[start_y:end_y, start_x:end_x]


def rotate_image(array, angle):
    height, width = array.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(array, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Calculate the distance between two points in 2D space using Euclidean distance formula.

    Args:
        p1 (tuple): Coordinates of the first point (x, y).
        p2 (tuple): Coordinates of the second point (x, y).

    Returns:
        float: The distance between the two points.
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def angle_between_points(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Calculate the angle (in degrees) formed by three points, with p1 as the vertex.

    Args:
        p1 (tuple): Coordinates of the first point (x, y).
        p2 (tuple): Coordinates of the second point (x, y).

    Returns:
        float: The angle (in degrees) formed by the three points.
    """
    angle_degrees = 0
    if p1 == p2:
        return math.inf

    p3 = (p2[0], p1[1])
    a, b, c = distance(p2, p3), distance(p1, p3), distance(p1, p2)

    if b == 0:
        if a > 0:
            angle_degrees = 90
        else:
            angle_degrees = 270
    elif c != 0:
        cos_a = (c ** 2 + b ** 2 - a ** 2) / (2 * c * b)
        angle_radians = math.acos(cos_a)
        angle_degrees = math.degrees(angle_radians)

        # Quadrant rectification
        # Q1 do nothing

        # Q2
        if p2[0] < p1[0] and p2[1] > p1[1]:
            angle_degrees = 180 - angle_degrees

        # Q3
        if p2[0] < p1[0] and p2[1] < p1[1]:
            angle_degrees = angle_degrees + 180

        # Q4
        if p2[0] > p1[0] and p2[1] < p1[1]:
            angle_degrees = -1 * angle_degrees

    return angle_degrees


def process_image(image: np.ndarray, window_name: str) -> None:
    """
    Display an image in a window and wait for mouse clicks.

    Args:
        image (numpy.ndarray): The input image.
        window_name (str): The name of the window.

    Returns:
        None
    """
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def get_triangle_coordinates(image_path: str, window_name: str) -> list[tuple[int, int]]:
    """
    Display an image and retrieve coordinates of three mouse clicks.

    Args:
        image_path (str): The path to the input image.
        window_name (str): The name of the window.

    Returns:
        list: Coordinates of three mouse clicks as (x, y) tuples.
    """
    image = cv2.imread(str(image_path))
    process_image(image, window_name)
    return clicked_coordinates[-3:]


def transformation_matrix(degrees: float, translation: tuple[int, int],scale_factor):
    beta = math.radians(degrees)
    tm_1 = [scale_factor*math.cos(beta), -1*scale_factor*math.sin(beta), translation[0]]
    tm_2 = [math.sin(beta), scale_factor * math.cos(beta), translation[1]]
    tm_3 = [0, 0, 1]
    return np.array([tm_1, tm_2, tm_3])


def main():
    # First we read the images
    path = 'images/brain/'
    image_path_a = path + 'dibuix_3.png'
    image_path_b = path + 'dibuix_3.png'

    image_a = cv2.imread(image_path_a)
    image_b = cv2.imread(image_path_b)

    a_coord_1, a_coord_2, a_coord_3 = get_triangle_coordinates(image_path_a, 'First Image')

    b_coord_1, b_coord_2, b_coord_3 = get_triangle_coordinates(image_path_b, 'Second image')

    # Second, we find the rotation value using the triangle

    angle_a2 = angle_between_points(a_coord_1, a_coord_3)
    angle_a3 = angle_between_points(a_coord_2, a_coord_3)

    angle_b2 = angle_between_points(b_coord_1, b_coord_3)
    angle_b3 = angle_between_points(b_coord_2, b_coord_3)

    # Angle difference, three must be the same but as its manual, we do this to minimize human error
    ddif_2 = angle_b2 - angle_a2
    ddif_3 = angle_b3 - angle_a3

    mean_ddif = (ddif_2 + ddif_3) / 2

    # Third, we find the scale factor
    a_tri_perimeter = distance(a_coord_1, a_coord_2) + distance(a_coord_1, a_coord_3) + distance(a_coord_2, a_coord_3)
    b_tri_perimeter = distance(b_coord_1, b_coord_2) + distance(b_coord_1, b_coord_3) + distance(b_coord_2, b_coord_3)

    scale_factor = a_tri_perimeter / b_tri_perimeter

    # Next, we rotate the image
    rotated_b = rotate_image(image_b, mean_ddif)

    # Then, we scale the image

    new_width = int(rotated_b.shape[1] * scale_factor)
    new_height = int(rotated_b.shape[0] * scale_factor)

    scaled_image = cv2.resize(rotated_b, (new_width, new_height))

    # We crop the image, so it matches size with the original
    cropped_image = crop_center(scaled_image, image_a.shape[1], image_a.shape[0])

    cv2.imwrite('final_image.jpg', cropped_image)

    cv2.imshow('Original Image', image_a)
    cv2.imshow('Rotated Image', cropped_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    process_image(cropped_image, "Click third flag")
    new_point = clicked_coordinates[-1]
    vector = (int(a_coord_3[0]) - int(new_point[0]), int(a_coord_3[1]) - int(new_point[1]))


    print(new_point)
    print(a_coord_3)
    print(vector)
    final_tm = transformation_matrix(mean_ddif, vector, scale_factor)
    b_point = np.array([[new_point[0]],
                        [new_point[1]],
                        [1]])

    original = np.dot(final_tm, b_point)

    print(original)
    print(a_coord_3)


if __name__ == "__main__":
    main()
