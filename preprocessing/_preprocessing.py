import pandas
import numpy
import os


class Preprocessing:
    full_dataset = None

    def __init__(
        self,
        from_file=True,
    ):
        if from_file:
            self.full_dataset = self.load_full_dataset()
        else:
            self.full_dataset = self.create_full_dataset()

    def series_to_supervised(
        self,
        dataset,
        window_size=1,
        n_out=1,
        dropnan=True,
    ):
        if type(dataset) is not pandas.core.frame.DataFrame:
            raise NameError('Dataset must pe a pandas DataFrame')

        number_of_features = 1 if type(dataset) is list else dataset.shape[1]
        features_names = dataset.columns
        cols = list()
        names = list()

        for i in range(window_size, 0, -1):
            cols.append(dataset.shift(i))
            names += [
                '{feature_name}-{time_step}'.format(
                    feature_name=features_names[j],
                    time_step=i,
                )
                for j in range(number_of_features)
            ]

        # for i in range(0, n_out):
        #     cols.append(dataset.shift(-i))
        #     if i == 0:
        #         names += [
        #             ('var%d(t)' % (j+1))
        #             for j in range(number_of_features)
        #         ]
        #     else:
        #         names += [
        #             ('var%d(t+%d)' % (j+1, i))
        #             for j in range(number_of_features)
        #         ]

        agg = pandas.concat(cols, axis=1)
        agg.columns = names

        if dropnan:
            agg.dropna(
                inplace=True,
            )

        return agg

    def reshape_dataset_to_model_input(
        self,
        dataset_values,
        number_of_samples,
        window_size,
        number_of_features,
    ):
        if type(dataset_values) == numpy.ndarray:
            raise NameError('Dataset must pe a pandas DataFrame')

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
