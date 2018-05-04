from . import _preprocessing


class AfulaSample(
    _preprocessing.Preprocessing,
):
    configuration_name = 'afula_sample'
    features = [
        'observedMaxTemp',
        'persist. value_1',
        'ec_1',
        'co_1',
        'c3_1',
        'oh_1',
        'ec_2',
        'co_2',
        'c3_2',
        'oh_2',
    ]

    def get_extracted_dataset(
        self,
        city='Afula',
    ):
        afula_dataset = self.full_dataset[self.full_dataset['city'] == city]
        afula_dataset = afula_dataset[self.features]
        afula_dataset.reset_index(
            drop=True,
            inplace=True,
        )

        return afula_dataset
