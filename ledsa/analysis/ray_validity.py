"""
Validity of the individual rays (image, LED) with respect to the additive pedestal in the LED search area.

Under dense smoke little LED light reaches the camera and the box sum consists mostly of the pedestal (stray light,
smoke glow, clipped noise). The plain reading then yields a lower bound of the optical depth instead of the optical
depth. Two criteria are evaluated per ray, both based on the side band pedestal P = N * B (sidebg_pedestal):

- raw reading (sum_col_val): saturated if S - P <= 0, or if b_raw = -ln(S / S0) underestimates the pedestal
  corrected optical depth b_corr = -ln((S - P) / (S0 - P0)) by more than the threshold, i.e.
  1 - b_raw / b_corr > threshold (only evaluated for b_corr > min_optical_depth, below the ratio is undetermined).
- side band corrected reading (sidebg_sum_col_val = S_corr): invalid if the floor applies (S - P <= floor * S), or if
  a relative error U_B of the pedestal shifts the optical depth b_sidebg = -ln(S_corr / S0_corr) by more than the
  threshold: delta_b = U_B * (P / S_corr + P0 / S0_corr) (image and reference term added, conservative),
  delta_b / b_sidebg > threshold (only for b_sidebg > min_optical_depth). Rays without side band pedestal (NaN)
  are invalid for the side band reading and are not marked as saturated in the raw reading.

The reference values S0, P0 and S0_corr are the means over the reference images, like the reference intensities of
the extinction coefficient calculation. Threshold 0.1, min_optical_depth 0.1 and U_B = 0.2 are set values (as in the
H2.0 evaluation), not derived ones; how well the side bands match the pedestal inside the box cannot be measured
directly.

The marking does not change the inversion: saturated rays still enter the extinction coefficient calculation and can
pull neighbouring layers.
"""
import os

import numpy as np
import pandas as pd

DEFAULT_VALIDITY_THRESHOLD = 0.1
DEFAULT_MIN_OPTICAL_DEPTH = 0.1
DEFAULT_SIDEBAND_UNCERTAINTY = 0.2
REQUIRED_COLUMNS = ['sum_col_val', 'sidebg_sum_col_val', 'sidebg_pedestal']


def mark_ray_validity(img_data: pd.DataFrame, ref_img_ids, floor: float,
                      threshold: float = DEFAULT_VALIDITY_THRESHOLD,
                      min_optical_depth: float = DEFAULT_MIN_OPTICAL_DEPTH,
                      sideband_uncertainty: float = DEFAULT_SIDEBAND_UNCERTAINTY) -> pd.DataFrame:
    """
    Mark saturated rays of the raw reading and invalid rays of the side band corrected reading.

    :param img_data: Step 3 data with index (img_id, led_id) and the columns sum_col_val, sidebg_sum_col_val and
        sidebg_pedestal (led_array_id is passed through if present).
    :type img_data: pd.DataFrame
    :param ref_img_ids: Ids of the reference images.
    :type ref_img_ids: Sequence[int]
    :param floor: Lower limit of sidebg_sum_col_val as fraction of sum_col_val used in step 3 (sideband_floor).
    :type floor: float
    :param threshold: Relative threshold of the optical depth for both criteria.
    :type threshold: float
    :param min_optical_depth: Optical depth below which the relative criteria are not evaluated.
    :type min_optical_depth: float
    :param sideband_uncertainty: Relative error U_B of the side band pedestal.
    :type sideband_uncertainty: float
    :return: DataFrame with index (img_id, led_id) and the columns led_array_id (if present), b_raw, b_corr,
        saturated_raw, b_sidebg, delta_b_sidebg, invalid_sidebg.
    :rtype: pd.DataFrame
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in img_data.columns]
    if missing:
        raise KeyError(f'Columns {missing} are missing. They are written by step 3 since the side band pedestal '
                       f'was added; run step 3 again.')
    img_ids = img_data.index.get_level_values('img_id')
    ref_data = img_data[np.isin(img_ids, ref_img_ids)]
    if ref_data.empty:
        raise ValueError(f'None of the reference images {list(ref_img_ids)} is available.')
    ref = ref_data.groupby(level='led_id')[REQUIRED_COLUMNS].mean()
    ref = ref.reindex(img_data.index.get_level_values('led_id'))

    s = img_data['sum_col_val'].to_numpy(dtype=np.float64)
    s_corr = img_data['sidebg_sum_col_val'].to_numpy(dtype=np.float64)
    pedestal = img_data['sidebg_pedestal'].to_numpy(dtype=np.float64)
    s0 = ref['sum_col_val'].to_numpy(dtype=np.float64)
    s0_corr = ref['sidebg_sum_col_val'].to_numpy(dtype=np.float64)
    pedestal0 = ref['sidebg_pedestal'].to_numpy(dtype=np.float64)

    signal, signal0 = s - pedestal, s0 - pedestal0
    with np.errstate(divide='ignore', invalid='ignore'):
        b_raw = np.log(s0 / s)
        b_corr = np.where((signal > 0) & (signal0 > 0), np.log(signal0 / signal), np.nan)
        saturated_raw = (signal <= 0) | ((b_corr > min_optical_depth) & (1 - b_raw / b_corr > threshold))
        b_sidebg = np.log(s0_corr / s_corr)
        delta_b = sideband_uncertainty * (pedestal / s_corr + pedestal0 / s0_corr)
        invalid_sidebg = ((signal <= floor * s) | ((b_sidebg > min_optical_depth) & (delta_b / b_sidebg > threshold))
                          | np.isnan(pedestal) | np.isnan(pedestal0))

    result = pd.DataFrame(index=img_data.index)
    if 'led_array_id' in img_data.columns:
        result['led_array_id'] = img_data['led_array_id'].to_numpy()
    result['b_raw'] = b_raw
    result['b_corr'] = b_corr
    result['saturated_raw'] = saturated_raw
    result['b_sidebg'] = b_sidebg
    result['delta_b_sidebg'] = delta_b
    result['invalid_sidebg'] = invalid_sidebg
    return result


def write_ray_validity(channel: int, ref_img_ids, floor: float, threshold: float = DEFAULT_VALIDITY_THRESHOLD,
                       min_optical_depth: float = DEFAULT_MIN_OPTICAL_DEPTH,
                       sideband_uncertainty: float = DEFAULT_SIDEBAND_UNCERTAINTY, path: str = '.') -> pd.DataFrame:
    """
    Mark the rays of one channel and write analysis/ray_validity_channel<channel>.csv.

    :param channel: Color channel.
    :type channel: int
    :param ref_img_ids: Ids of the reference images, or a callable that returns them for the step 3 data (DataFrame
        with index (img_id, led_id)).
    :param floor: Lower limit of sidebg_sum_col_val as fraction of sum_col_val used in step 3.
    :param threshold: See mark_ray_validity.
    :param min_optical_depth: See mark_ray_validity.
    :param sideband_uncertainty: See mark_ray_validity.
    :param path: Root directory of the LEDSA simulation.
    :type path: str
    :return: The marking, see mark_ray_validity.
    :rtype: pd.DataFrame
    """
    from ledsa.core.file_handling import read_hdf
    img_data = read_hdf(channel, path=path)
    if callable(ref_img_ids):
        ref_img_ids = ref_img_ids(img_data)
    validity = mark_ray_validity(img_data, ref_img_ids, floor, threshold, min_optical_depth, sideband_uncertainty)
    out = validity.copy()
    out['saturated_raw'] = out['saturated_raw'].astype(int)
    out['invalid_sidebg'] = out['invalid_sidebg'].astype(int)
    out_file = os.path.join(path, 'analysis', f'ray_validity_channel{channel}.csv')
    header = (f'# ray validity, channel {channel}: reference images {list(map(int, ref_img_ids))}, '
              f'threshold {threshold}, min_optical_depth {min_optical_depth}, '
              f'sideband_uncertainty {sideband_uncertainty}, sideband_floor {floor}\n')
    with open(out_file, 'w', encoding='utf-8') as file:
        file.write(header)
        out.reset_index().to_csv(file, index=False, float_format='%.6g', lineterminator='\n')
    print(f'{out_file} written: {100 * validity.saturated_raw.mean():.1f} % of the rays saturated in the raw '
          f'reading, {100 * validity.invalid_sidebg.mean():.1f} % invalid in the side band corrected reading.')
    return validity
