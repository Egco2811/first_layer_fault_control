import os


def count_images(directory):
    """
    Count the number of image files in the specified directory.

    Args:
        directory (str): The path to the directory containing images.

    Returns:
        int: The count of image files in the directory.
    """
    count = 0
    for root, _, files in os.walk(directory):
        count += len([f for f in files if f.endswith('.jpg')])
    return count
