from . import _preprocessing


class AfulaSample(
    _preprocessing.Preprocessing,
):
    configuration_name = 'afula_sample'
    features = [
        'observedMaxTemp',
        'persist. value_maxtemp_1',
        'ec_maxtemp_2',
        'co_maxtemp_2',
        'c3_maxtemp_2',
        'oh_maxtemp_2',
    ]

    def get_extracted_dataset(
        self,
    ):
        afula_dataset = self.full_dataset[self.full_dataset['city'] == 'Afula']
        afula_dataset = afula_dataset[self.features]
        afula_dataset.reset_index(
            drop=True,
            inplace=True,
        )

        return afula_dataset
