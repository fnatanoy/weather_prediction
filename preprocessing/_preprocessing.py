import pandas
import numpy
import os


class Preprocessing:
    full_dataset = None

    def __init__(
        self,
        file_name='',
    ):
        if file_name is '':
            self.full_dataset = self.create_full_dataset()
        else:
            dataset_path = os.path.join(
                'data',
                file_name + '.pkl',
            )
            self.full_dataset = pandas.read_pickle(
                path=dataset_path,
            )

    def series_to_supervised(
        self,
        dataset,
        window_size=2,
        n_out=1,
        dropnan=True,
    ):
        if type(dataset) is not pandas.core.frame.DataFrame:
            raise NameError('Dataset must pe a pandas DataFrame')

        number_of_features = dataset.shape[1]
        features_names = dataset.columns
        cols = list()
        names = list()

        for i in range(window_size - 1, -1, -1):
            cols.append(dataset.shift(i))
            names += [
                '{feature_name}_t-{time_step}'.format(
                    feature_name=features_names[j],
                    time_step=i,
                )
                for j in range(number_of_features)
            ]

        dataset_series = pandas.concat(cols, axis=1)
        dataset_series.columns = names

        if dropnan:
            dataset_series.dropna(
                inplace=True,
            )

        return dataset_series

    def reshape_dataset_to_model_input(
        self,
        dataset_values,
        number_of_samples,
        window_size,
        number_of_features,
    ):
        if type(dataset_values) is not numpy.ndarray:
            raise NameError('Dataset must be values')

        return dataset_values.reshape(
            (
                number_of_samples,
                window_size,
                number_of_features,
            ),
        )

    def get_extracted_dataset(
        self,
    ):
        raise NotImplemented
