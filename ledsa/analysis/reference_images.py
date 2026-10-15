from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_REF_IMG_TOLERANCE = 0.02


def get_ref_img_ids(img_ids: np.ndarray, num_ref_imgs: Optional[int] = 10,
                    ref_img_indices: Optional[List[int]] = None, brightness: Optional[pd.Series] = None,
                    tolerance: float = DEFAULT_REF_IMG_TOLERANCE,
                    num_candidates: Optional[int] = None) -> np.ndarray:
    """
    Reference image ids as used for the reference intensities of the extinction coefficient calculation: the given
    ref_img_indices, otherwise, if brightness is given, the automatic selection by brightness (see
    select_ref_img_ids_by_brightness), otherwise all images with img_id <= num_ref_imgs.

    :param img_ids: Available image ids.
    :type img_ids: np.ndarray
    :param num_ref_imgs: Number of reference images if ref_img_indices is None.
    :type num_ref_imgs: int or None
    :param ref_img_indices: Explicit reference image ids.
    :type ref_img_indices: list[int] or None
    :param brightness: Box sums with index (img_id, led_id) for the automatic selection, None to disable it.
    :type brightness: pd.Series or None
    :param tolerance: See select_ref_img_ids_by_brightness.
    :type tolerance: float
    :param num_candidates: See select_ref_img_ids_by_brightness.
    :type num_candidates: int or None
    :return: Sorted reference image ids that are available.
    :rtype: np.ndarray
    """
    img_ids = np.unique(np.asarray(img_ids))
    if ref_img_indices is not None:
        return img_ids[np.isin(img_ids, ref_img_indices)]
    if brightness is not None:
        selected = select_ref_img_ids_by_brightness(brightness, num_ref_imgs, tolerance, num_candidates)
        return img_ids[np.isin(img_ids, selected)]
    return img_ids[img_ids <= num_ref_imgs]


def select_ref_img_ids_by_brightness(brightness: pd.Series, num_ref_imgs: int,
                                     tolerance: float = DEFAULT_REF_IMG_TOLERANCE,
                                     num_candidates: Optional[int] = None) -> List[int]:
    """
    Select the reference images automatically by their brightness.

    Single images at the beginning of a series can be darker or brighter than the others (e.g. the first image of a
    series of some cameras), which biases the reference intensities. The brightness of each candidate image relative
    to the others is the median over all LEDs of the box sum divided by the median of that LED over the candidate
    images. Images within the tolerance are valid; the first num_ref_imgs valid images are selected.

    The candidates are the first num_candidates images (default 2 * num_ref_imgs). They should lie before the smoke
    reaches the LED arrays: images with smoke are only excluded if they are darker than the tolerance, and since the
    brightness is relative to the median over the candidates, more than half of the candidates must be free of smoke.

    :param brightness: Box sums (sum_col_val) with index (img_id, led_id).
    :type brightness: pd.Series
    :param num_ref_imgs: Number of reference images to select.
    :type num_ref_imgs: int
    :param tolerance: Maximum relative deviation of the brightness from 1.
    :type tolerance: float
    :param num_candidates: Number of images at the beginning of the series that are considered.
    :type num_candidates: int or None
    :return: Selected image ids.
    :rtype: list[int]
    :raises ValueError: If no candidate image is within the tolerance.
    """
    if num_candidates is None:
        num_candidates = 2 * num_ref_imgs
    img_ids = np.unique(brightness.index.get_level_values('img_id'))
    candidates = img_ids[:num_candidates]
    table = brightness[brightness.index.get_level_values('img_id').isin(candidates)].unstack('img_id')
    relative = table.div(table.median(axis=1), axis=0).median()
    valid = relative.index[(relative - 1).abs() <= tolerance]
    selected = [int(img_id) for img_id in valid[:num_ref_imgs]]
    excluded = {int(img_id): round(float(value), 3) for img_id, value in relative.items()
                if abs(value - 1) > tolerance}
    print(f'Brightness of images {int(candidates[0])}-{int(candidates[-1])} relative to their median: '
          f'{relative.min():.3f}-{relative.max():.3f}' + (f', excluded {excluded}' if excluded else ''))
    if not selected:
        raise ValueError(f'No reference image within the brightness tolerance {tolerance} among the images '
                         f'{list(map(int, candidates))}.')
    if len(selected) < num_ref_imgs:
        print(f'Warning: only {len(selected)} instead of {num_ref_imgs} reference images within the brightness '
              f'tolerance: {selected}')
    return selected
