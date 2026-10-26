import numpy as np
import pytest

import ledsa.core.image_reading as image_reading
from ledsa.core.image_integrity import CorruptedImageError


class TestRawExtensionRouting:
    @pytest.fixture
    def raw_reader_stub(self, monkeypatch):
        calls = []

        def stub(filename, channel, black_level_per_channel=False):
            calls.append(filename)
            return np.zeros((4, 4))

        monkeypatch.setattr(image_reading, '_read_channel_data_from_raw_file', stub)
        return calls

    @pytest.mark.parametrize('extension', ['.CR2', '.CR3', '.NEF', '.ARW', '.DNG', '.nef', '.dng'])
    def test_raw_formats_are_routed_to_raw_reader(self, raw_reader_stub, extension):
        image_reading.read_channel_data_from_img(f'img_0001{extension}', channel=0)
        assert raw_reader_stub == [f'img_0001{extension}']

    @pytest.mark.parametrize('reader', [image_reading.read_channel_data_from_img,
                                        image_reading.read_channel_data_and_mask_from_img])
    def test_unsupported_format_raises(self, reader):
        with pytest.raises(CorruptedImageError, match='unsupported file extension'):
            reader('img_0001.xyz', channel=0)
