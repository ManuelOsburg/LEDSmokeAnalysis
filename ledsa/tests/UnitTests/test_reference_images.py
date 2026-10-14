from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ledsa.analysis.ExtinctionCoefficients as ec_module
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients
from ledsa.analysis.reference_images import get_ref_img_ids, select_ref_img_ids_by_brightness

NUM_LEDS = 50


def make_brightness(factors, seed=0):
    """Box sums with index (img_id, led_id): per LED a random level, per image a brightness factor and 0.2 % noise."""
    rng = np.random.default_rng(seed)
    levels = rng.uniform(1e4, 6e4, NUM_LEDS)
    rows = []
    for img_id, factor in enumerate(factors, start=1):
        values = levels * factor * (1 + rng.normal(0, 0.002, NUM_LEDS))
        rows += [(img_id, led_id, value) for led_id, value in enumerate(values)]
    data = pd.DataFrame(rows, columns=['img_id', 'led_id', 'sum_col_val']).set_index(['img_id', 'led_id'])
    return data['sum_col_val']


class TestSelectByBrightness:
    def test_dark_first_image_is_excluded(self):
        # like the first image of the J1 series in H2.0 (42 % brightness) and a 2.5 % darker image 4
        factors = np.ones(20)
        factors[0] = 0.42
        factors[3] = 0.975
        selected = select_ref_img_ids_by_brightness(make_brightness(factors), 10)
        assert selected == [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]

    def test_all_images_within_tolerance(self):
        assert select_ref_img_ids_by_brightness(make_brightness(np.ones(20)), 10) == list(range(1, 11))

    def test_tolerance(self):
        factors = np.ones(20)
        factors[1] = 0.97
        assert 2 in select_ref_img_ids_by_brightness(make_brightness(factors), 10, tolerance=0.05)
        assert 2 not in select_ref_img_ids_by_brightness(make_brightness(factors), 10, tolerance=0.02)

    def test_only_candidates_are_considered(self, capsys):
        # images 5 to 20 are dimmed by smoke; the brightness is relative to the median over the candidates, so the
        # majority of the candidates must be free of smoke
        factors = np.concatenate([np.ones(4), np.linspace(0.9, 0.3, 16)])
        selected = select_ref_img_ids_by_brightness(make_brightness(factors), 3)  # candidates 1-6
        assert selected == [1, 2, 3]
        selected = select_ref_img_ids_by_brightness(make_brightness(factors), 5, num_candidates=6)
        assert selected == [1, 2, 3, 4]
        assert 'Warning: only 4 instead of 5' in capsys.readouterr().out

    def test_no_valid_image_raises(self):
        factors = np.array([0.5, 0.6, 1.4, 1.5])
        with pytest.raises(ValueError, match='No reference image'):
            select_ref_img_ids_by_brightness(make_brightness(factors), 3)

    def test_get_ref_img_ids(self):
        factors = np.ones(20)
        factors[0] = 0.42
        brightness = make_brightness(factors)
        img_ids = np.arange(1, 21)
        np.testing.assert_array_equal(get_ref_img_ids(img_ids, 3), [1, 2, 3])
        np.testing.assert_array_equal(get_ref_img_ids(img_ids, 3, brightness=brightness), [2, 3, 4])
        # explicit indices take precedence over the automatic selection
        np.testing.assert_array_equal(get_ref_img_ids(img_ids, 3, [1, 5], brightness=brightness), [1, 5])


class DummyCoefficients(ExtinctionCoefficients):
    def calc_coefficients_of_img(self, rel_intensities):
        return rel_intensities


class TestExtinctionCoefficientsReference:
    def make(self, monkeypatch, **kwargs):
        factors = np.ones(20)
        factors[0] = 0.42
        brightness = make_brightness(factors)
        img_data = brightness.to_frame()
        img_data['led_array_id'] = 0
        monkeypatch.setattr(ec_module, 'read_hdf', lambda channel, path='.': img_data.copy())
        coefficients = DummyCoefficients(SimpleNamespace(channel=0, path='.'), num_ref_imgs=5, **kwargs)
        coefficients.calculated_img_data = img_data[['led_array_id', 'sum_col_val']]
        return coefficients, brightness

    def test_default_uses_first_images(self, monkeypatch):
        coefficients, brightness = self.make(monkeypatch)
        with pytest.warns(UserWarning):  # image 1 is dark, the existing normalisation check warns
            coefficients.calc_and_set_ref_intensities()
        expected = brightness[brightness.index.get_level_values('img_id') <= 5].groupby(level='led_id').mean()
        np.testing.assert_array_equal(coefficients.ref_intensities, expected.to_numpy())
        assert coefficients.ref_img_indices is None

    def test_automatic_selection(self, monkeypatch):
        coefficients, brightness = self.make(monkeypatch, auto_ref_imgs=True)
        coefficients.calc_and_set_ref_intensities()
        assert coefficients.ref_img_indices == [2, 3, 4, 5, 6]
        selected = brightness.index.get_level_values('img_id').isin([2, 3, 4, 5, 6])
        np.testing.assert_array_equal(coefficients.ref_intensities,
                                      brightness[selected].groupby(level='led_id').mean().to_numpy())
        assert 'Ref_img_indices: [2, 3, 4, 5, 6]' in str(coefficients)

    def test_explicit_indices_take_precedence(self, monkeypatch):
        coefficients, _ = self.make(monkeypatch, auto_ref_imgs=True, ref_img_indices=[7, 8])
        coefficients.calc_and_set_ref_intensities()
        assert coefficients.ref_img_indices == [7, 8]
