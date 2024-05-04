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


def triangle_size(c1: tuple[int, int], c2: tuple[int, int], c3: tuple[int, int]) -> list[float]:
    """
    Calculate the size of each side of a triangle formed by three points.

    Args:
        c1 (tuple): Coordinates of the first point (x, y).
        c2 (tuple): Coordinates of the second point (x, y).
        c3 (tuple): Coordinates of the third point (x, y).

    Returns:
        list: A list containing the lengths of the triangle sides, sorted in descending order.
    """
    return sorted([distance(c1, c2), distance(c1, c3), distance(c2, c3)], reverse=True)


def angle_between_points(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Calculate the angle (in degrees) formed by three points, with p1 as the vertex.

    Args:
        p1 (tuple): Coordinates of the first point (x, y).
        p2 (tuple): Coordinates of the second point (x, y).

    Returns:
        float: The angle (in degrees) formed by the three points.
    """
    p3 = (p1[0], p2[1])
    a, b, c = distance(p2, p3), distance(p1, p3), distance(p1, p2)
    cos_a = (c ** 2 + b ** 2 - a ** 2) / (2 * c * b)
    angle_radians = math.acos(cos_a)
    angle_degrees = math.degrees(angle_radians)
    if p2[0] > p1[0]:
        angle_degrees = 180 - angle_degrees
    return angle_degrees


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


def process_images(image_path_1: str, image_path_2: str) -> np.ndarray:
    """
    Process two images to rotate, zoom, and crop the second image based on user input.

    Args:
        image_path_1 (str): The path to the first (reference) image.
        image_path_2 (str): The path to the second (distorted) image.

    Returns:
        numpy.ndarray: The processed and transformed second image.
    """
    if len(clicked_coordinates) < 3:
        print('1st image: Click at least 3 times.')
        return np.array([])
    else:
        coord_1, coord_2, coord_3 = get_triangle_coordinates(image_path_1, 'Primera Imatge')

        positions_1 = [coord_1, coord_2]
        reference_degree = angle_between_points(coord_2, coord_1)
        clicked_coordinates.clear()

        if len(clicked_coordinates) < 3:
            print('2nd image: Click at least 3 times.')
            return np.array([])
        else:
            coord_1, coord_2, coord_3 = get_triangle_coordinates(image_path_2, 'Segona imatge')

            positions_2 = [coord_1, coord_2]
            rotated_degree = angle_between_points(coord_2, coord_1)
            difference = rotated_degree - reference_degree
            scale_factor = distance(positions_1[0], positions_1[1]) / distance(positions_2[0], positions_2[1])

            og_image = cv2.imread(str(image_path_1))
            og_height, og_width = og_image.shape[:2]

            dis_image = cv2.imread(str(image_path_2))

            rotation_matrix = cv2.getRotationMatrix2D(coord_2, difference, 1)
            rotated_image = cv2.warpAffine(dis_image, rotation_matrix, (og_width, og_height))

            new_width = int(rotated_image.shape[1] * scale_factor)
            new_height = int(rotated_image.shape[0] * scale_factor)

            zoomed_image = cv2.resize(rotated_image, (new_width, new_height))
            final_image = crop_center(zoomed_image, og_image.shape[1], og_image.shape[0])

            return final_image


def main():
    """
    Entry point of the program.
    Processes two images and displays the final transformed image.
    """

    path = 'images/brain/'
    image_path_1 = path + 'dibuix_1.png'
    image_path_2 = path + 'dibuix_2_zoomedout.png'

    if len(clicked_coordinates) < 3:
        print('1st image: Click at least 3 times.')
        return
    else:
        coord_1, coord_2, coord_3 = get_triangle_coordinates(image_path_1, 'Primera Imatge')

        positions_1 = [coord_1, coord_2]
        reference_degree = angle_between_points(coord_2, coord_1)
        clicked_coordinates.clear()

        if len(clicked_coordinates) < 3:
            print('2nd image: Click at least 3 times.')
            return
        else:
            coord_1, coord_2, coord_3 = get_triangle_coordinates(image_path_2, 'Segona imatge')

            positions_2 = [coord_1, coord_2]
            rotated_degree = angle_between_points(coord_2, coord_1)
            difference = rotated_degree - reference_degree
            zoom_factor = distance(positions_1[0], positions_1[1]) / distance(positions_2[0], positions_2[1])

            og_image = cv2.imread(image_path_1)
            og_height, og_width = og_image.shape[:2]

            dis_image = cv2.imread(image_path_2)

            rotation_matrix = cv2.getRotationMatrix2D(coord_2, difference, 1)
            rotated_image = cv2.warpAffine(dis_image, rotation_matrix, (og_width, og_height))

            new_width = int(rotated_image.shape[1]*zoom_factor)
            new_height = int(rotated_image.shape[0]*zoom_factor)

            zoomed_image = cv2.resize(rotated_image, (new_width, new_height))
            final_image = crop_center(zoomed_image, og_image.shape[1], og_image.shape[0])
            print(type(zoomed_image))
            cv2.imshow('Original Image', og_image)
            cv2.imshow('Rotated Image', final_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
