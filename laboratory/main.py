"""
Closest Pair of Points Finder
==============================

This module implements a divide and conquer algorithm to find the closest pair
of 2D points from a dataset.

It reads coordinates from a `.txt` input file, computes the closest pair by
Euclidean distance, and prints the result along with the execution time.

Examples
--------
Input file format:

    Line 1: x1,x2,x3,...
    Line 2: y1,y2,y3,...

Author: Sergio Andrés Majé Franco
Date: 2025-05-19
Institution: Fundación Universitaria Internacional La Rioja
Course: Diseño de Algoritmos Avanzados
Assignment: Laboratorio #1: Par más cercano
"""


import time
import math


def read_points(file_path: str) -> list[tuple[float, float]]:
    """
    Reads a file containing two lines: the first with x-values, the second
    with y-values.

    Parameters
    ----------
    file_path : str
        Path to the input file with coordinates.

    Returns
    -------
    list of tuple of float
        List of 2D points represented as tuples (x, y).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If x and y lists have different lengths.
    """
    with open(file_path, encoding="utf-8") as file:
        x_line = file.readline().strip()
        y_line = file.readline().strip()

    x_coords = list(map(float, x_line.split(",")))
    y_coords = list(map(float, y_line.split(",")))

    if len(x_coords) != len(y_coords):
        raise ValueError("The number of x and y coordinates must be equal.")

    return list(zip(x_coords, y_coords))


def euclidean_distance(
    p1: tuple[float, float],
    p2: tuple[float, float]
) -> float:
    """
    Computes the Euclidean distance between two 2D points.

    This function calculates the straight-line (as-the-crow-flies) distance
    between two points in a 2D Cartesian plane using the Pythagorean theorem.

    Parameters
    ----------
    p1 : tuple of float
        First point.
    p2 : tuple of float
        Second point.

    Returns
    -------
    float
        Euclidean distance between p1 and p2.

    Notes
    -----
    The Euclidean distance formula in 2D is:

        d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

    This function is used extensively in both the brute-force and the
    divide-and-conquer approaches to determine proximity between two points.
    """
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def brute_force(
    points: list[tuple[float, float]]
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """
    Finds the closest pair of points using the brute-force method.

    This function checks the Euclidean distance between every possible
    pair of points in the input list to determine the closest pair. It
    is used in the base case of the divide and conquer approach, typically
    when the number of points is less than or equal to 3.

    Parameters
    ----------
    points : list of tuple of float
        List of 2D points.

    Returns
    -------
    tuple
        Minimum distance and the corresponding pair of points.

    Notes
    -----
    The brute-force approach compares all possible pairs of points.

    Time complexity: O(n²), where n is the number of points.

    This function is only efficient for small inputs and is used
    as the base case in the recursive divide and conquer algorithm.
    """
    min_dist = float("inf")
    closest_pair = (points[0], points[1])

    for i, point1 in enumerate(points):
        for point2 in points[i + 1:]:
            dist = euclidean_distance(point1, point2)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (point1, point2)

    return min_dist, closest_pair[0], closest_pair[1]


def closest_pair_recursive(
    sorted_by_x: list[tuple[float, float]],
    sorted_by_y: list[tuple[float, float]]
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """
    Recursively finds the closest pair of points using the divide and
    conquer approach.

    This function assumes that the input points are pre-sorted by both x
    and y coordinates. It divides the problem into two halves, recursively
    finds the closest pair in each half, and then checks whether a closer
    pair exists across the dividing line.

    Parameters
    ----------
    sorted_by_x : list of tuple of float
        Points sorted by x-coordinate.
    sorted_by_y : list of tuple of float
        Points sorted by y-coordinate.

    Returns
    -------
    tuple
        Minimum distance and the closest pair of points.

    Notes
    -----
    This function implements the standard divide and conquer algorithm for
    the closest pair of points problem with a time complexity of O(n log n),
    where n is the number of input points.

    The three main steps are:

    1. **Divide**:
       - Split the points into two halves using the midpoint of the
         x-sorted list.

    2. **Conquer**:
       - Recursively compute the closest pair in each half.

    3. **Combine**:
       - Check for possible closer pairs across the dividing line within
         a strip of width `2 * min_distance` centered at the midpoint.
       - Only up to 6 neighbors ahead in the y-sorted strip need to be
         checked due to geometric constraints in the 2D plane.
    """
    num_points = len(sorted_by_x)

    # Base case: solve by brute-force when small input
    if num_points <= 3:
        return brute_force(sorted_by_x)

    # Step 1: Divide
    mid = num_points // 2
    mid_point = sorted_by_x[mid]

    left_x = sorted_by_x[:mid]
    right_x = sorted_by_x[mid:]

    left_y = [p for p in sorted_by_y if p[0] <= mid_point[0]]
    right_y = [p for p in sorted_by_y if p[0] > mid_point[0]]

    # Step 2: Conquer
    dist_left, p1_left, p2_left = closest_pair_recursive(left_x, left_y)
    dist_right, p1_right, p2_right = closest_pair_recursive(right_x, right_y)

    min_dist = dist_left
    closest_pair = (p1_left, p2_left)

    if dist_right < dist_left:
        min_dist = dist_right
        closest_pair = (p1_right, p2_right)

    # Step 3: Combine - check across the split line
    strip = [p for p in sorted_by_y if abs(p[0] - mid_point[0]) < min_dist]

    for i, point1 in enumerate(strip):
        for point2 in strip[i + 1:i + 7]:
            dist = euclidean_distance(point1, point2)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (point1, point2)

    return min_dist, closest_pair[0], closest_pair[1]


def find_closest_pair(
    points: list[tuple[float, float]]
) -> tuple[float, tuple[float, float], tuple[float, float], float]:
    """
    Finds the closest pair of points and measures the execution time.

    Parameters
    ----------
    points : list of tuple of float
        List of 2D points.

    Returns
    -------
    tuple
        Minimum distance, point 1, point 2, and execution time in seconds.
    """
    start_time = time.time()
    sorted_by_x = sorted(points, key=lambda p: p[0])
    sorted_by_y = sorted(points, key=lambda p: p[1])

    min_distance, point1, point2 = closest_pair_recursive(
        sorted_by_x, sorted_by_y
    )
    end_time = time.time()

    return min_distance, point1, point2, end_time - start_time


def main() -> None:
    """
    Main function to read input, compute closest pair and display results.

    Notes
    -----
    Ensure the input file has the following format:
    - Line 1: comma-separated x-values
    - Line 2: comma-separated y-values

    Example
    -------
    1,2,3,10
    4,5,6,12
    """
    inputs = ["datos_100.txt", "datos_1000.txt", "datos_10000.txt"]
    input_file = inputs[0]  # Change this to test different files

    try:
        points = read_points(input_file)
        (
            distance_value, point_a, point_b, elapsed_time
        ) = find_closest_pair(points)

        print(f"Closest pair: {point_a} and {point_b}")
        print(f"Minimum distance: {distance_value:.6f}")
        print(f"Execution time: {elapsed_time:.6f} seconds")

    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
