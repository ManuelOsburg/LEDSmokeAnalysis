import argparse
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

import ledsa.core.version
from ledsa.analysis.StackedExtinctionCoefficients import StackedExtinctionCoefficients
from ledsa.core.parser_arguments_declaration import add_parser_arguments_data_extraction
from ledsa.data_extraction.step_1_functions import find_search_areas


class TestMaxNumLedsWarning:
    @staticmethod
    def image_with_spots(num_spots):
        img = np.zeros((40, 40 * num_spots), dtype=np.float32)
        for i in range(num_spots):
            img[20, 20 + 40 * i] = 100.0
        return img

    def test_warning_when_search_stops_at_max_num_leds(self, capsys):
        search_areas, _ = find_search_areas(self.image_with_spots(5), search_area_radius=5,
                                            pixel_value_percentile=99.0, max_n_leds=3)
        assert len(search_areas) == 3
        assert 'Warning' in capsys.readouterr().out

    def test_no_warning_when_all_leds_are_found(self, capsys):
        search_areas, _ = find_search_areas(self.image_with_spots(3), search_area_radius=5,
                                            pixel_value_percentile=99.0, max_n_leds=3)
        assert len(search_areas) == 3
        assert 'Warning' not in capsys.readouterr().out


class TestStep3FastWithChannel:
    def test_s3_fast_with_channel_does_not_start_the_fit(self, tmp_path, monkeypatch):
        import ledsa.core.parser_arguments_run as run
        created = []

        class FakeDataExtractor:
            def __init__(self, build_experiment_infos=True, channels=(0,), fit_leds=True):
                created.append((list(channels), fit_leds))

            def setup_step3(self):
                pass

            def process_image_data(self):
                pass

        monkeypatch.chdir(tmp_path)
        (tmp_path / 'config.ini').write_text('[DEFAULT]\n', encoding='utf-8')
        monkeypatch.setattr(run, 'DataExtractor', FakeDataExtractor)
        parser = add_parser_arguments_data_extraction(argparse.ArgumentParser())
        run.run_data_extraction_arguments(parser.parse_args(['-s3_fast', '-g']))
        assert created == [([1], False)]
        created.clear()
        run.run_data_extraction_arguments(parser.parse_args(['-g']))
        assert created == [([1], True)]


class TestTimeAlignmentReport:
    def make_stacked(self, tmp_path, monkeypatch):
        stacked = StackedExtinctionCoefficients.__new__(StackedExtinctionCoefficients)
        stacked.config = SimpleNamespace(output_path=tmp_path, time_sync_tolerance=1.0,
                                         get_simulation_path=lambda sim_idx: tmp_path / f'sim{sim_idx}',
                                         get_num_simulations=lambda: 2)
        stacked.led_array_id = 0
        stacked.led_array_mapping = {0: 0, 1: 0}
        stacked._sub_solvers = [SimpleNamespace(experiment=None, average_images=False, calculated_img_data=None)] * 2
        monkeypatch.setattr(StackedExtinctionCoefficients, '_load_time_map',
                            staticmethod(lambda experiment, average_images, img_data=None: {1: 0.0, 2: 5.0}))
        return stacked

    def test_report_is_utf8_and_an_empty_report_is_written_again(self, tmp_path, monkeypatch):
        stacked = self.make_stacked(tmp_path, monkeypatch)
        report = tmp_path / 'time_alignment_led_array_0.txt'
        report.write_bytes(b'')  # left over from an interrupted run
        stacked._compute_time_alignment()
        content = report.read_bytes()
        assert len(content) > 0
        assert 'Δ'.encode('utf-8') in content
        content.decode('utf-8')
        assert not list(tmp_path.glob('*.tmp'))

    def test_existing_report_is_kept(self, tmp_path, monkeypatch):
        stacked = self.make_stacked(tmp_path, monkeypatch)
        report = tmp_path / 'time_alignment_led_array_0.txt'
        report.write_text('existing report\n', encoding='utf-8')
        stacked._compute_time_alignment()
        assert report.read_text(encoding='utf-8') == 'existing report\n'


class TestVersion:
    def test_version_of_the_running_code(self):
        ledsa.core.version.get_ledsa_version.cache_clear()
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(ledsa.core.version.__file__)))
        pyproject = os.path.join(os.path.dirname(package_dir), 'pyproject.toml')
        if not os.path.exists(pyproject):
            pytest.skip('not running from a source checkout')
        with open(pyproject, encoding='utf-8') as file:
            expected = re.search(r'^version\s*=\s*"([^"]+)"', file.read(), re.M).group(1)
        assert ledsa.core.version.get_ledsa_version().startswith(expected)

    def test_distribution_installed_elsewhere_is_ignored(self, tmp_path, monkeypatch):
        fake = SimpleNamespace(version='0.9.4', locate_file=lambda name: tmp_path / 'site-packages' / name)
        monkeypatch.setattr(ledsa.core.version, 'distribution', lambda name: fake)
        assert ledsa.core.version._get_distribution_version(str(tmp_path / 'checkout' / 'ledsa')) is None
        assert ledsa.core.version._get_distribution_version(str(tmp_path / 'site-packages' / 'ledsa')) == '0.9.4'
