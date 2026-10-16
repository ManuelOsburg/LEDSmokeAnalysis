from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ledsa.analysis.ExtinctionCoefficients as ec_module
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients
from ledsa.analysis.ExtinctionCoefficientsLinear import ExtinctionCoefficientsLinear
from ledsa.analysis.ExtinctionCoefficientsNonLinear import ExtinctionCoefficientsNonLinear
from ledsa.analysis.reference_images import (evaluate_ref_img_candidates, get_ref_img_ids, load_experiment_times,
                                             read_ref_img_options, select_ref_img_ids_by_brightness)

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
        selected = select_ref_img_ids_by_brightness(make_brightness(factors), 5, num_candidates=6, num_ref_img_min=3)
        assert selected == [1, 2, 3, 4]
        assert 'Warning: only 4 instead of 5' in capsys.readouterr().out

    def test_no_valid_image_raises(self):
        factors = np.array([0.5, 0.6, 1.4, 1.5])
        with pytest.raises(ValueError, match='Only 0 valid reference images'):
            select_ref_img_ids_by_brightness(make_brightness(factors), 3)

    def test_fewer_than_minimum_raises(self):
        # 4 valid of 6 candidates; the default minimum is min(5, num_ref_imgs)
        factors = np.concatenate([np.ones(4), np.linspace(0.9, 0.3, 16)])
        with pytest.raises(ValueError, match='Only 4 valid reference images .* at least 5 are required'):
            select_ref_img_ids_by_brightness(make_brightness(factors), 5, num_candidates=6)
        with pytest.raises(ValueError, match='Only 2 valid reference images .* at least 3 are required'):
            select_ref_img_ids_by_brightness(make_brightness(np.array([1.0, 1.0, 0.5, 1.5])), 2, num_candidates=4,
                                             num_ref_img_min=3)

    def test_unsorted_index(self):
        # index in lexicographic file order (1, 10, 100, 11, ...) as after concatenating files found by glob
        factors = np.ones(100)
        factors[0] = 0.42
        brightness = make_brightness(factors)
        order = sorted(range(1, 101), key=str)
        unsorted = pd.concat([brightness.xs(i, level='img_id', drop_level=False) for i in order])
        expected = list(range(2, 12))
        assert select_ref_img_ids_by_brightness(brightness, 10) == expected
        assert select_ref_img_ids_by_brightness(unsorted, 10) == expected
        string_ids = unsorted.copy()
        string_ids.index = string_ids.index.set_levels(string_ids.index.levels[0].astype(str), level='img_id')
        assert select_ref_img_ids_by_brightness(string_ids, 10) == expected

    def test_partial_occlusion(self, capsys):
        # 15 % of the LEDs of image 4 are dimmed by 50 %: the median over all LEDs is inconspicuous
        brightness = make_brightness(np.ones(20))
        occluded = brightness.copy()
        leds = np.arange(int(0.15 * NUM_LEDS))
        occluded.loc[[(4, led) for led in leds]] *= 0.5
        assert 4 in select_ref_img_ids_by_brightness(occluded, 10, led_fraction=1.0)  # criterion disabled
        selected = select_ref_img_ids_by_brightness(occluded, 10)
        assert 4 not in selected
        assert selected == [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        out = capsys.readouterr().out
        assert 'image 4 excluded: partial occlusion' in out

    def test_smoke_after_max_time(self, capsys):
        # series starting shortly before the ignition: images 1-6 clean (t < 0), then smoke. Without a time limit the
        # normalisation over the mostly smoky candidates makes the clean images look too bright.
        factors = np.concatenate([np.ones(6), np.linspace(0.97, 0.6, 14)])
        brightness = make_brightness(factors)
        times = pd.Series(5.0 * np.arange(20) - 27.0, index=np.arange(1, 21))  # -27 s ... +68 s
        with pytest.raises(ValueError):
            select_ref_img_ids_by_brightness(brightness, 10)
        assert 'image 1 excluded: brightness' in capsys.readouterr().out
        selected = select_ref_img_ids_by_brightness(brightness, 10, experiment_times=times, max_time=0.0,
                                                    num_ref_img_min=5)
        assert selected == [1, 2, 3, 4, 5, 6]
        out = capsys.readouterr().out
        assert 'image 1 excluded' not in out
        assert 'time +3 s > +0 s' in out
        assert 'Warning: only 6 instead of 10' in out

    def test_max_time_without_times_raises(self):
        with pytest.raises(ValueError, match='experiment times'):
            select_ref_img_ids_by_brightness(make_brightness(np.ones(20)), 10, max_time=0.0)

    def test_normalisation_falls_back_to_all_candidates(self):
        # fewer than 3 candidates before max_time: normalised over all candidates, only the time criterion excludes
        brightness = make_brightness(np.ones(20))
        times = pd.Series(5.0 * np.arange(20) - 7.0, index=np.arange(1, 21))  # only images 1 and 2 before 0 s
        selected = select_ref_img_ids_by_brightness(brightness, 10, experiment_times=times, max_time=0.0,
                                                    num_ref_img_min=2)
        assert selected == [1, 2]

    def test_evaluation_table(self):
        factors = np.ones(20)
        factors[0] = 0.42
        evaluation = evaluate_ref_img_candidates(make_brightness(factors), 20)
        assert list(evaluation.index) == list(range(1, 21))
        assert not evaluation.loc[1, 'valid'] and evaluation.loc[1, 'reason'].startswith('brightness 0.42')
        assert evaluation.loc[2:, 'valid'].all()


class TestReadOptions:
    def test_old_config_falls_back_to_defaults(self):
        import configparser
        config = configparser.ConfigParser()
        config.read_string('[DEFAULT]\nnum_ref_images = 10\nauto_ref_images = True\nref_img_tolerance = 0.05\n')
        assert read_ref_img_options(config) == dict(auto_ref_imgs=True, ref_img_tolerance=0.05,
                                                    num_ref_img_candidates=None, ref_img_led_deviation=0.2,
                                                    ref_img_led_fraction=0.02, ref_img_max_time=None,
                                                    num_ref_img_min=None)

    def test_written_config(self, tmp_path, monkeypatch):
        from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False, auto_ref_images=True, ref_img_max_time=0, num_ref_img_min=3,
                           ref_img_led_fraction=0.05)
        options = read_ref_img_options(ConfigDataAnalysis())
        assert options['auto_ref_imgs'] is True
        assert options['ref_img_max_time'] == 0.0
        assert options['num_ref_img_min'] == 3
        assert options['ref_img_led_fraction'] == 0.05
        assert options['ref_img_led_deviation'] == 0.2

    def test_load_experiment_times(self, tmp_path):
        (tmp_path / 'analysis').mkdir()
        (tmp_path / 'analysis' / 'image_infos_analysis.csv').write_text(
            '#ID,Name,Time[s],Experiment_Time[s]\n1,a.CR3,10:00:00,-20.0\n2,b.CR3,10:00:05,-15.0\n')
        times = load_experiment_times(tmp_path)
        assert times.to_dict() == {1: -20.0, 2: -15.0}

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

    def test_options_are_passed_to_the_selection(self, monkeypatch):
        times = pd.Series(5.0 * np.arange(20) - 22.0, index=np.arange(1, 21))  # images 1-5 before 0 s
        monkeypatch.setattr(ec_module, 'load_experiment_times', lambda path, average_images=False: times)
        for solver in (ExtinctionCoefficientsLinear, ExtinctionCoefficientsNonLinear):
            factors = np.ones(20)
            factors[0] = 0.42
            img_data = make_brightness(factors).to_frame()
            img_data['led_array_id'] = 0
            monkeypatch.setattr(ec_module, 'read_hdf', lambda channel, path='.': img_data.copy())
            experiment = SimpleNamespace(channel=0, path='.', layers=SimpleNamespace(amount=3))
            coefficients = solver(experiment, num_ref_imgs=5, auto_ref_imgs=True, ref_img_max_time=0.0,
                                  num_ref_img_min=3)
            coefficients.calculated_img_data = img_data[['led_array_id', 'sum_col_val']]
            coefficients.calc_and_set_ref_intensities()
            assert coefficients.ref_img_indices == [2, 3, 4, 5]
