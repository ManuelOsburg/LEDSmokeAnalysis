import numpy as np
import pandas as pd
import pytest

from ledsa.analysis.ray_validity import mark_ray_validity, write_ray_validity
from ledsa.analysis.reference_images import get_ref_img_ids

FLOOR = 0.05


def make_data(rays):
    """Two reference images (S0 = 1000, P0 = 10) plus one image with the given (S, P) per LED."""
    rows = []
    for img_id in (1, 2):
        for led_id in range(len(rays)):
            rows.append((img_id, led_id, 0, 1000.0, 10.0))
    for led_id, (s, p) in enumerate(rays):
        rows.append((3, led_id, 0, s, p))
    data = pd.DataFrame(rows, columns=['img_id', 'led_id', 'led_array_id', 'sum_col_val', 'sidebg_pedestal'])
    data['sidebg_sum_col_val'] = np.maximum(data.sum_col_val - data.sidebg_pedestal, FLOOR * data.sum_col_val)
    return data.set_index(['img_id', 'led_id'])


class TestMarkRayValidity:
    def mark(self, rays):
        return mark_ray_validity(make_data(rays), [1, 2], FLOOR).loc[3]

    def test_light_smoke_is_valid(self):
        result = self.mark([(500.0, 10.0)])
        assert not result.saturated_raw.iloc[0]
        assert not result.invalid_sidebg.iloc[0]
        assert result.b_raw.iloc[0] == pytest.approx(-np.log(0.5))
        assert result.b_corr.iloc[0] == pytest.approx(-np.log(490 / 990))

    def test_pedestal_dominated_ray_is_saturated(self):
        # b_raw = 3.22 underestimates b_corr = 4.60 by 30 %
        result = self.mark([(40.0, 30.0)])
        assert result.saturated_raw.iloc[0]
        # delta_b = 0.2 * (30 / 10 + 10 / 990) = 0.60 > 0.1 * 4.60
        assert result.delta_b_sidebg.iloc[0] == pytest.approx(0.2 * (30 / 10 + 10 / 990))
        assert result.invalid_sidebg.iloc[0]

    def test_pedestal_above_sum(self):
        result = self.mark([(20.0, 25.0)])
        assert result.saturated_raw.iloc[0]
        assert np.isnan(result.b_corr.iloc[0])
        assert result.invalid_sidebg.iloc[0]

    def test_floor_makes_sidebg_invalid(self):
        result = self.mark([(30.0, 29.5)])
        assert result.invalid_sidebg.iloc[0]
        assert result.b_sidebg.iloc[0] == pytest.approx(-np.log(FLOOR * 30 / 990))

    def test_small_optical_depth_is_not_evaluated(self):
        # b_corr = 0.052 < 0.1: relative criteria are undetermined, the ray counts as valid
        result = self.mark([(950.0, 10.0)])
        assert not result.saturated_raw.iloc[0]
        assert not result.invalid_sidebg.iloc[0]

    def test_threshold_of_raw_criterion(self):
        # choose P such that 1 - b_raw / b_corr is just above / below 0.1 for b_raw = -ln(0.2)
        s = 200.0
        b_raw = -np.log(s / 1000)
        for factor, expected in [(1.12, True), (1.10, False)]:
            signal = 990 * np.exp(-b_raw * factor)
            result = self.mark([(s, s - signal)])
            assert result.saturated_raw.iloc[0] == expected

    def test_missing_pedestal(self):
        result = self.mark([(500.0, np.nan)])
        assert not result.saturated_raw.iloc[0]
        assert result.invalid_sidebg.iloc[0]

    def test_reference_uses_mean_of_reference_images(self):
        data = make_data([(500.0, 10.0)])
        data.loc[(2, 0), 'sum_col_val'] = 1200.0
        data.loc[(2, 0), 'sidebg_sum_col_val'] = 1190.0
        result = mark_ray_validity(data, [1, 2], FLOOR).loc[3]
        assert result.b_raw.iloc[0] == pytest.approx(-np.log(500 / 1100))

    def test_old_data_without_side_band_columns(self):
        data = make_data([(500.0, 10.0)]).drop(columns=['sidebg_pedestal'])
        with pytest.raises(KeyError, match='step 3'):
            mark_ray_validity(data, [1, 2], FLOOR)


class TestRefImgIds:
    def test_num_ref_imgs(self):
        np.testing.assert_array_equal(get_ref_img_ids(np.array([3, 1, 2, 2, 5]), 2), [1, 2])

    def test_explicit_indices(self):
        np.testing.assert_array_equal(get_ref_img_ids(np.arange(1, 10), 2, [2, 4, 11]), [2, 4])


def test_write_ray_validity(tmp_path):
    data = make_data([(500.0, 10.0), (40.0, 30.0)]).reset_index()
    (tmp_path / 'analysis' / 'channel0').mkdir(parents=True)
    data.to_hdf(tmp_path / 'analysis' / 'channel0' / 'all_parameters.h5', key='channel_values', format='table')
    write_ray_validity(0, lambda img_data: get_ref_img_ids(img_data.index.get_level_values("img_id"), 2), FLOOR,
                       path=str(tmp_path))
    with open(tmp_path / 'analysis' / 'ray_validity_channel0.csv') as file:
        header = file.readline()
        table = pd.read_csv(file)
    assert 'reference images [1, 2]' in header
    assert list(table.columns) == ['img_id', 'led_id', 'led_array_id', 'b_raw', 'b_corr', 'saturated_raw', 'b_sidebg',
                                   'delta_b_sidebg', 'invalid_sidebg']
    image = table[table.img_id == 3].set_index('led_id')
    assert image.saturated_raw.tolist() == [0, 1]
    assert image.invalid_sidebg.tolist() == [0, 1]
