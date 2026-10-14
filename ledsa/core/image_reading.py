import os
from typing import Tuple

import exiv2
import numpy as np
import rawpy
from matplotlib import pyplot as plt


def read_channel_data_from_img(filename: str, channel: int) -> np.ndarray:
    """
    Returns a 2D array of the image for a single color channel.
    8bit is default range for JPG. For RAW files the Bayer array is returned as a 2D array where
    all channel values except the selected channel are masked.

    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel to process.
    :type channel: int
    :return: A 2D array containing the processed image data for one channel.
    :rtype: np.ndarray
    """
    extension = os.path.splitext(filename)[-1]
    if extension in ['.JPG', '.JPEG', '.jpg', '.jpeg', '.PNG', '.png']:
        channel_array = _read_channel_data_from_img_file(filename, channel)
    elif extension in ['.CR2', '.CR3', '.NEF', '.ARW', '.DNG']:
        channel_array = _read_channel_data_from_raw_file(filename, channel)
    else:
        raise ValueError(f"Unsupported image format '{extension}' of file {filename}.")
    return channel_array

def read_img_array_from_img(filename: str, channel: int) -> np.ndarray:
    """
    Returns a 2D array of the image for all color channels.


    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel used for calculating black level values.
    :type channel: int
    :return: A 2D array containing the processed image data for all color channels.
    :rtype: np.ndarray
    """
    extension = os.path.splitext(filename)[-1]
    if extension in ['.JPG', '.JPEG', '.jpg', '.jpeg', '.PNG', '.png']:
        img_array = _read_grayscale_img_array_from_img_file(filename)
    elif extension in ['.CR2', '.CR3', '.NEF', '.ARW', '.DNG']:
        img_array, _ = _read_img_array_from_raw_file(filename, channel)
    else:
        raise ValueError(f"Unsupported image format '{extension}' of file {filename}.")
    return img_array


def _read_channel_data_from_img_file(filename: str, channel: int) -> np.ndarray:
    """
    Reads an image file and extracts a single color channel.

    :param filename: The file path of the image to be read.
    :type filename: str
    :param channel: The index of the color channel to extract (0 for Red, 1 for Green, 2 for Blue in RGB images).
    :type channel: int
    :return: A 2D numpy array containing the data of the specified color channel from the image.
    :rtype: np.ndarray
    """
    img_array = _read_img_array_from_img_file(filename)
    return img_array[:, :, channel]


def _read_channel_data_from_raw_file(filename: str, channel: int) -> np.ndarray:
    """
    Extracts a single color channel from a RAW image file.

    :param filename: The file path of the RAW image to be read.
    :type filename: str
    :param channel: The index of the color channel to extract. The Bayer array typically contains two green channels (1 and 3),
                    one red (0), and one blue (2).
    :type channel: int
    :return: A 2D numpy array representing the extracted channel, with all other channel values masked or set to zero.
    :rtype: np.ndarray
    """
    img_array, filter_array = _read_img_array_from_raw_file(filename, channel)
    if channel == 0 or channel == 2:
        channel_array = np.where(filter_array == channel, img_array, 0)
    elif channel == 1:
        channel_array = np.where((filter_array == 1) | (filter_array == 3), img_array, 0)
    return channel_array


def read_channel_data_and_mask_from_img(filename: str, channel: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the same 2D channel array as read_channel_data_from_img together with a boolean mask of the
    pixels that belong to the selected color channel.

    For RAW files the mask is taken from the Bayer filter pattern (raw_colors_visible) and not from the pixel
    values, because pixels of the selected channel can be 0 as well (noise clipped at 0 after the black level
    subtraction). For JPG/PNG files every pixel belongs to the channel.

    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel to process.
    :type channel: int
    :return: The channel array and the boolean channel mask of the same shape.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    extension = os.path.splitext(filename)[-1]
    if extension in ['.JPG', '.JPEG', '.jpg', '.jpeg', '.PNG', '.png']:
        channel_array = _read_channel_data_from_img_file(filename, channel)
        channel_mask = np.ones(channel_array.shape, dtype=bool)
    elif extension in ['.CR2', '.CR3', '.NEF', '.ARW', '.DNG']:
        img_array, filter_array = _read_img_array_from_raw_file(filename, channel)
        channel_mask = get_bayer_channel_mask(filter_array, channel)
        channel_array = np.where(channel_mask, img_array, 0)
    else:
        raise ValueError(f"Unsupported image format '{extension}' of file {filename}.")
    return channel_array, channel_mask


def get_bayer_channel_mask(filter_array: np.ndarray, channel: int) -> np.ndarray:
    """
    Returns a boolean mask of the Bayer filter positions that belong to the given color channel.
    The Bayer array contains two green channels (1 and 3), one red (0) and one blue (2).

    :param filter_array: Bayer filter pattern of the raw image (rawpy raw_colors_visible).
    :type filter_array: np.ndarray
    :param channel: The color channel (0 red, 1 green, 2 blue).
    :type channel: int
    :return: Boolean mask, True where the pixel belongs to the channel.
    :rtype: np.ndarray
    """
    if channel == 1:
        return (filter_array == 1) | (filter_array == 3)
    return filter_array == channel


def _read_img_array_from_raw_file(filename: str, channel: int) -> np.ndarray:
    # TODO: channel is only relevant for black level, consider individually!
    with rawpy.imread(filename) as raw:
        data = raw.raw_image_visible.copy()
        filter_array = raw.raw_colors_visible
        black_level = raw.black_level_per_channel[channel]
        white_level = raw.white_level
    img_array = data.astype(np.int16) - black_level
    img_array = (img_array * (white_level / (white_level - black_level))).astype(np.int16)
    img_array = np.clip(img_array, 0, white_level)
    return img_array, filter_array

def _read_img_array_from_img_file(filename: str) -> np.ndarray:
    img_array = plt.imread(filename)
    return img_array

def _read_grayscale_img_array_from_img_file(filename: str) -> np.ndarray:
    img_array = plt.imread(filename)
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = np.dot(img_array[..., :3], weights).astype(np.uint8)
    return gray


def get_exif_entry(filename: str, tag: str) -> str:
    """
    Retrieves the EXIF metadata entry from an image.

    :param filename: The path of the image file to read.
    :type filename: str
    :param tag: The EXIF metadata tag to fetch.
    :type tag: str
    :return: The value(s) associated with the given EXIF tag.
    :rtype: str
    :raises KeyError: If the EXIF tag is not found in the image metadata.
    """
    img = exiv2.ImageFactory.open(filename)
    img.readMetadata()
    exiv_data = img.exifData()
    full_tag = f'Exif.Photo.{tag}'
    try:
        return exiv_data[full_tag].print()
    except KeyError:
        print("No EXIF metadata found")
        exit(1)
