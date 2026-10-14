import numpy as np
import pytest

import ledsa.core.image_reading as image_reading

WHITE_LEVEL = 16383
BLACK_LEVELS = [510, 512, 514, 512]  # per Bayer channel R, G1, B, G2


class FakeRaw:
    """Stands in for rawpy.RawPy: RGGB pattern, constant signal per channel plus noise around the black level."""

    def __init__(self, signal=(0.0, 0.0, 0.0, 0.0), noise_sigma=4.0, shape=(200, 200), seed=0):
        rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
        self.raw_colors_visible = np.where(rows % 2 == 0, np.where(cols % 2 == 0, 0, 1),
                                           np.where(cols % 2 == 0, 3, 2)).astype(np.uint8)
        rng = np.random.default_rng(seed)
        black = np.asarray(BLACK_LEVELS, dtype=float)[self.raw_colors_visible]
        signal = np.asarray(signal, dtype=float)[self.raw_colors_visible]
        self.raw_image_visible = np.rint(black + signal + rng.normal(0.0, noise_sigma, shape)).astype(np.uint16)
        self.black_level_per_channel = BLACK_LEVELS
        self.white_level = WHITE_LEVEL

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def fake_raw(monkeypatch):
    def install(**kwargs):
        raw = FakeRaw(**kwargs)
        monkeypatch.setattr(image_reading.rawpy, 'imread', lambda filename: raw)
        return raw
    return install


class TestDefaultRawReading:
    def test_default_is_unchanged(self, fake_raw):
        raw = fake_raw(signal=(30, 40, 50, 40))
        img_array, filter_array = image_reading._read_img_array_from_raw_file('x.CR3', 1)
        black = BLACK_LEVELS[1]
        expected = raw.raw_image_visible.astype(np.int16) - black
        expected = (expected * (WHITE_LEVEL / (WHITE_LEVEL - black))).astype(np.int16)
        expected = np.clip(expected, 0, WHITE_LEVEL)
        np.testing.assert_array_equal(img_array, expected)
        assert img_array.dtype == expected.dtype
        np.testing.assert_array_equal(filter_array, raw.raw_colors_visible)


class TestBlackLevelPerChannel:
    def test_each_pixel_gets_the_black_level_of_its_channel(self, fake_raw):
        raw = fake_raw(signal=(30, 40, 50, 40))
        img_array, _ = image_reading._read_img_array_from_raw_file('x.CR3', 1, black_level_per_channel=True)
        black = np.asarray(BLACK_LEVELS, dtype=float)[raw.raw_colors_visible]
        expected = (raw.raw_image_visible - black) * (WHITE_LEVEL / (WHITE_LEVEL - black))
        np.testing.assert_allclose(img_array, expected)
        assert img_array.dtype == np.float64

    def test_dark_pixels_are_not_clipped(self, fake_raw):
        fake_raw(signal=(0, 0, 0, 0))
        img_array, _ = image_reading._read_img_array_from_raw_file('x.CR3', 1, black_level_per_channel=True)
        assert img_array.min() < 0
        # noise around the black level averages to zero, for every channel
        assert abs(img_array.mean()) < 0.1

    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_pedestal_bias_of_the_channel_sum(self, fake_raw, channel):
        # dark channel with a small signal of 1 count: clipping rectifies the noise and adds a pedestal to
        # the channel sum, the float path without clipping keeps the true mean
        fake_raw(signal=(1.0, 1.0, 1.0, 1.0), noise_sigma=4.0)
        default = image_reading.read_channel_data_from_img('x.CR3', channel)
        per_channel, mask = image_reading.read_channel_data_and_mask_from_img('x.CR3', channel,
                                                                              black_level_per_channel=True)
        scale = WHITE_LEVEL / (WHITE_LEVEL - BLACK_LEVELS[channel])
        assert per_channel[mask].mean() == pytest.approx(1.0 * scale, abs=0.15)
        assert default[mask].mean() > 1.0 * scale + 0.5
        assert np.all(per_channel[~mask] == 0)

    def test_green_channel_uses_both_green_black_levels(self, fake_raw):
        raw = fake_raw(signal=(0, 100, 0, 100), noise_sigma=0.0)
        data = image_reading.read_channel_data_from_img('x.CR3', 1, black_level_per_channel=True)
        green = (raw.raw_colors_visible == 1) | (raw.raw_colors_visible == 3)
        scale_g1 = WHITE_LEVEL / (WHITE_LEVEL - BLACK_LEVELS[1])
        scale_g2 = WHITE_LEVEL / (WHITE_LEVEL - BLACK_LEVELS[3])
        np.testing.assert_allclose(data[raw.raw_colors_visible == 1], 100 * scale_g1)
        np.testing.assert_allclose(data[raw.raw_colors_visible == 3], 100 * scale_g2)
        assert np.all(data[~green] == 0)
