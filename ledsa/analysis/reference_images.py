import configparser as cp
import os
from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_REF_IMG_TOLERANCE = 0.02
DEFAULT_REF_IMG_LED_DEVIATION = 0.2
DEFAULT_REF_IMG_LED_FRACTION = 0.02
DEFAULT_NUM_REF_IMG_MIN = 5
# minimum number of candidates within ref_img_max_time to normalise the LEDs only over these
MIN_NORMALISATION_IMGS = 3


def get_ref_img_ids(img_ids: np.ndarray, num_ref_imgs: Optional[int] = 10,
                    ref_img_indices: Optional[List[int]] = None, brightness: Optional[pd.Series] = None,
                    tolerance: float = DEFAULT_REF_IMG_TOLERANCE,
                    num_candidates: Optional[int] = None, **selection_options) -> np.ndarray:
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
    :param selection_options: Further keyword arguments of select_ref_img_ids_by_brightness (led_deviation,
        led_fraction, experiment_times, max_time, num_ref_img_min).
    :return: Sorted reference image ids that are available.
    :rtype: np.ndarray
    """
    img_ids = np.unique(np.asarray(img_ids))
    if ref_img_indices is not None:
        return img_ids[np.isin(img_ids, ref_img_indices)]
    if brightness is not None:
        selected = select_ref_img_ids_by_brightness(brightness, num_ref_imgs, tolerance, num_candidates,
                                                    **selection_options)
        return img_ids[np.isin(img_ids, selected)]
    return img_ids[img_ids <= num_ref_imgs]


def evaluate_ref_img_candidates(brightness: pd.Series, num_candidates: int,
                                tolerance: float = DEFAULT_REF_IMG_TOLERANCE,
                                led_deviation: float = DEFAULT_REF_IMG_LED_DEVIATION,
                                led_fraction: float = DEFAULT_REF_IMG_LED_FRACTION,
                                experiment_times: Optional[pd.Series] = None,
                                max_time: Optional[float] = None) -> pd.DataFrame:
    """
    Evaluate the first num_candidates images (by img_id) as reference images.

    For every LED the box sum of each candidate is divided by the median of that LED over the candidates (only over
    the candidates with experiment time <= max_time if there are at least MIN_NORMALISATION_IMGS of them, so that
    candidates with smoke do not shift the normalisation). A candidate is valid if

    - its brightness, the median of these ratios over all LEDs, deviates by at most tolerance from 1,
    - at most a fraction led_fraction of the LEDs deviates by more than led_deviation (partial occlusion, e.g. a
      person passing in front of the LED arrays, does not show in the median),
    - its experiment time is <= max_time (if given).

    :param brightness: Box sums (sum_col_val) with index (img_id, led_id). The order of the index does not matter.
    :type brightness: pd.Series
    :param num_candidates: Number of images at the beginning of the series that are considered.
    :type num_candidates: int
    :param tolerance: Maximum relative deviation of the brightness from 1.
    :type tolerance: float
    :param led_deviation: Relative deviation from which an LED counts as deviating.
    :type led_deviation: float
    :param led_fraction: Maximum fraction of deviating LEDs.
    :type led_fraction: float
    :param experiment_times: Experiment time [s] per img_id, required if max_time is given.
    :type experiment_times: pd.Series or None
    :param max_time: Latest experiment time [s] of a reference image, None for no limit.
    :type max_time: float or None
    :return: One row per candidate (index img_id, ascending) with the columns brightness, deviating_fraction,
        experiment_time, valid and reason (empty for valid candidates).
    :rtype: pd.DataFrame
    """
    table = _brightness_table(brightness)
    candidates = table.columns[:num_candidates]
    table = table[candidates]

    times = pd.Series(np.nan, index=candidates, dtype=float)
    in_time = pd.Series(True, index=candidates)
    if max_time is not None:
        if experiment_times is None:
            raise ValueError('ref_img_max_time is set, but no experiment times are available.')
        times = pd.Series(experiment_times).reindex(candidates).astype(float)
        if times.isna().any():
            raise ValueError(f'No experiment time for the candidate images {list(map(int, times.index[times.isna()]))}.')
        in_time = times <= max_time
    normalisation = candidates[in_time.to_numpy()] if in_time.sum() >= MIN_NORMALISATION_IMGS else candidates

    relative = table.div(table[normalisation].median(axis=1), axis=0)
    img_brightness = relative.median()
    deviating_fraction = ((relative - 1).abs() > led_deviation).sum() / relative.notna().sum()

    reasons = []
    for img_id in candidates:
        reason = []
        if abs(img_brightness[img_id] - 1) > tolerance:
            reason.append(f'brightness {img_brightness[img_id]:.3f}')
        if deviating_fraction[img_id] > led_fraction:
            reason.append(f'partial occlusion ({100 * deviating_fraction[img_id]:.0f} % of the LEDs deviate by '
                          f'> {100 * led_deviation:.0f} %)')
        if not in_time[img_id]:
            reason.append(f'time {times[img_id]:+.0f} s > {max_time:+.0f} s')
        reasons.append(', '.join(reason))
    result = pd.DataFrame({'brightness': img_brightness, 'deviating_fraction': deviating_fraction,
                           'experiment_time': times, 'reason': reasons}, index=candidates)
    result['valid'] = result['reason'] == ''
    result.index.name = 'img_id'
    return result


def select_ref_img_ids_by_brightness(brightness: pd.Series, num_ref_imgs: int,
                                     tolerance: float = DEFAULT_REF_IMG_TOLERANCE,
                                     num_candidates: Optional[int] = None,
                                     led_deviation: float = DEFAULT_REF_IMG_LED_DEVIATION,
                                     led_fraction: float = DEFAULT_REF_IMG_LED_FRACTION,
                                     experiment_times: Optional[pd.Series] = None,
                                     max_time: Optional[float] = None,
                                     num_ref_img_min: Optional[int] = None) -> List[int]:
    """
    Select the reference images automatically by their brightness.

    Single images at the beginning of a series can be darker or brighter than the others (e.g. the first image of a
    series of some cameras), or LEDs can be partly occluded, which biases the reference intensities. The candidates
    are the first num_candidates images (default 2 * num_ref_imgs) and are evaluated with
    evaluate_ref_img_candidates; the first num_ref_imgs valid ones are selected.

    The reference images must lie before the smoke reaches the LED arrays. Without max_time, images with smoke are
    only excluded if they are darker than the tolerance, and more than half of the candidates must be free of smoke,
    because the brightness is relative to the median over the candidates. If the series starts shortly before the
    ignition, set max_time (e.g. 0, the ignition, if Experiment_Time counts from the ignition).

    :param brightness: Box sums (sum_col_val) with index (img_id, led_id). The order of the index does not matter.
    :type brightness: pd.Series
    :param num_ref_imgs: Number of reference images to select.
    :type num_ref_imgs: int
    :param tolerance: See evaluate_ref_img_candidates.
    :type tolerance: float
    :param num_candidates: Number of images at the beginning of the series that are considered.
    :type num_candidates: int or None
    :param led_deviation: See evaluate_ref_img_candidates.
    :type led_deviation: float
    :param led_fraction: See evaluate_ref_img_candidates.
    :type led_fraction: float
    :param experiment_times: Experiment time [s] per img_id, required if max_time is given.
    :type experiment_times: pd.Series or None
    :param max_time: Latest experiment time [s] of a reference image, None for no limit.
    :type max_time: float or None
    :param num_ref_img_min: Minimum number of valid reference images, default min(5, num_ref_imgs).
    :type num_ref_img_min: int or None
    :return: Selected image ids, ascending.
    :rtype: list[int]
    :raises ValueError: If fewer than num_ref_img_min candidates are valid.
    """
    if num_candidates is None:
        num_candidates = 2 * num_ref_imgs
    if num_ref_img_min is None:
        num_ref_img_min = min(DEFAULT_NUM_REF_IMG_MIN, num_ref_imgs)
    evaluation = evaluate_ref_img_candidates(brightness, num_candidates, tolerance, led_deviation, led_fraction,
                                             experiment_times, max_time)
    selected = [int(img_id) for img_id in evaluation.index[evaluation['valid']][:num_ref_imgs]]
    excluded = evaluation[~evaluation['valid']]
    print(f'Reference image candidates {int(evaluation.index[0])}-{int(evaluation.index[-1])}: brightness relative '
          f'to their median {evaluation.brightness.min():.3f}-{evaluation.brightness.max():.3f}')
    for img_id, row in excluded.iterrows():
        print(f'  image {int(img_id)} excluded: {row.reason}')
    if len(selected) < num_ref_img_min:
        raise ValueError(f'Only {len(selected)} valid reference images {selected} among the images '
                         f'{list(map(int, evaluation.index))}, at least {num_ref_img_min} are required. Check the '
                         f'excluded images above; adjust ref_img_max_time, num_ref_img_candidates or set '
                         f'ref_img_indices explicitly.')
    if len(selected) < num_ref_imgs:
        print(f'Warning: only {len(selected)} instead of {num_ref_imgs} valid reference images: {selected}')
    return selected


def load_experiment_times(path='.', average_images: bool = False) -> pd.Series:
    """
    Experiment times per img_id from analysis/image_infos_analysis.csv (or _avg.csv), with img_id = row number + 1
    as in the stacked analysis.

    :param path: Root directory of the LEDSA simulation.
    :param average_images: Use the file of the averaged images.
    :type average_images: bool
    :return: Experiment time [s] with index img_id.
    :rtype: pd.Series
    """
    info_file = 'image_infos_analysis_avg.csv' if average_images else 'image_infos_analysis.csv'
    times = pd.read_csv(os.path.join(path, 'analysis', info_file))['Experiment_Time[s]'].to_numpy(dtype=float)
    return pd.Series(times, index=pd.Index(np.arange(1, len(times) + 1), name='img_id'))


def read_ref_img_options(config: cp.ConfigParser, section: str = 'DEFAULT') -> dict:
    """
    Read the options of the automatic reference image selection from a config (config_analysis.ini or
    config_stacked.ini). Missing options (older files) fall back to the defaults.

    :param config: The config.
    :type config: configparser.ConfigParser
    :param section: Section of the options.
    :type section: str
    :return: Keyword arguments for ExtinctionCoefficients: auto_ref_imgs, ref_img_tolerance, num_ref_img_candidates,
        ref_img_led_deviation, ref_img_led_fraction, ref_img_max_time, num_ref_img_min.
    :rtype: dict
    """
    def value(option, convert, default):
        raw = config.get(section, option, fallback='None')
        return default if raw is None or str(raw).strip() == 'None' else convert(raw)

    return dict(auto_ref_imgs=config.getboolean(section, 'auto_ref_images', fallback=False),
                ref_img_tolerance=value('ref_img_tolerance', float, DEFAULT_REF_IMG_TOLERANCE),
                num_ref_img_candidates=value('num_ref_img_candidates', int, None),
                ref_img_led_deviation=value('ref_img_led_deviation', float, DEFAULT_REF_IMG_LED_DEVIATION),
                ref_img_led_fraction=value('ref_img_led_fraction', float, DEFAULT_REF_IMG_LED_FRACTION),
                ref_img_max_time=value('ref_img_max_time', float, None),
                num_ref_img_min=value('num_ref_img_min', int, None))


def _brightness_table(brightness: pd.Series) -> pd.DataFrame:
    """Box sums as table with one row per LED and one column per image, columns sorted by the numeric img_id."""
    values = brightness.to_numpy()
    img_ids = brightness.index.get_level_values('img_id').astype(int)
    led_ids = brightness.index.get_level_values('led_id')
    series = pd.Series(values, index=pd.MultiIndex.from_arrays([img_ids, led_ids], names=['img_id', 'led_id']))
    table = series.unstack('img_id')
    return table.reindex(columns=np.sort(table.columns.to_numpy()))
