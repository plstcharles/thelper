import mock
import pytest

from thelper.data.geo.utils import parse_raster_metadata

# whole package marked
pytestmark = pytest.mark.geo


class MockDatasetSentinel2(object):
    _driver = "SENTINEL2"
    _meta = {
        "SUBDATASETS": {
            "SUBDATASET_1_NAME": "{}:some-path:some-resolution:some-crs".format(_driver),
            "SUBDATASET_2_NAME": "{}:other-path:other-resolution:other-crs".format(_driver)
        }
    }

    def GetDriver(self):
        class Driver(object):
            @property
            def ShortName(self):
                return MockDatasetSentinel2._driver
        return Driver()

    def GetMetadata(self, meta):
        return self._meta.get(meta, {})

    def GetRasterBand(self, band):
        if isinstance(band, int) and band > 0:
            return "fake-but-valid-get-raster-band"
        return None


def test_parse_raster_metadata():
    raster_meta = {
        "path": "S2A_MSIL1C_RANDOM.zip",
        "bands": [2, 3, 4, 8],
        "subdataset": 2,
    }
    with mock.patch("gdal.OpenShared", return_value=MockDatasetSentinel2()) as mock_gdal_open:
        parse_raster_metadata(raster_meta)
        assert mock_gdal_open.called_once
    assert raster_meta["path"] == MockDatasetSentinel2._meta["SUBDATASETS"]["SUBDATASET_2_NAME"]
    assert raster_meta["reader"] == ""
