from . import _preprocessing


class DatasetConfiguration1(
    _preprocessing.Preprocessing,
):
    configuration_name = 'config1'

    def get_extracted_dataset(
        self,
    ):
        return ''
