import os
import sklearn
import sklearn.preprocessing
import preprocessing


class TrainModel:
    models_architecture_name = ''
    cities = [
        'Afula',
        'Ashdod',
        'Beer Sheva',
        'Bet Shean',
        'Elat',
        'En Gedi',
        'Haifa',
        'Jerusalem',
        'Lod',
        'Mizpe Ramon',
        'Nazareth',
        'Qazrin',
        'Tel Aviv',
        'Tiberias',
        'Zefat',
    ]

    def __init__(
        self,
    ):
        pass

    def run(
        self,
        dataset_config,
        hyperparameters,
    ):
        preprocessor = preprocessing.datasets_structures[dataset_config](
            file_name='train_maxTemp',
        )
        for city in self.cities:
            print(city)
            dataset = preprocessor.get_extracted_dataset(
                city=city,
            ) / 45
            target = dataset['observedMaxTemp']
            target = target[hyperparameters['window_size'] - 1:]

            dataset.drop(
                ['observedMaxTemp'],
                axis=1,
                inplace=True,
            )
            number_of_features = dataset.shape[1]

            dataset_series = preprocessor.series_to_supervised(
                dataset=dataset,
                window_size=hyperparameters['window_size'],
            )

            # scaler_samples = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
            # scaler_target = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
            # samples = scaler_samples.fit_transform(scaler_samples)
            # target = scaler_target.fit_transform(scaler_target)

            samples = preprocessor.reshape_dataset_to_model_input(
                dataset_values=dataset_series.values,
                number_of_samples=dataset_series.shape[0],
                window_size=hyperparameters['window_size'],
                number_of_features=number_of_features,
            )
            self.compile_model(
                input_shape=samples.shape[1:],
                hyperparameters=hyperparameters,
            )

            self.train_model(
                samples=samples,
                target=target,
                hyperparameters=hyperparameters,
            )
            self.save_model()

        # self.evaluating_model(
        #     testing_samples=samples_pad_sequences[testing_indexes, :],
        #     targets=dataset[labels].values[testing_indexes, :],
        #     hyperparameters=hyperparameters,
        # )

    def split_training_testing(
        self,
        dataset,
        hyperparameters,
    ):
        pass

    def save_model(
        self,
    ):
        model_json = self.model.to_json()
        with open(
            os.path.join(
                'trained_models',
                'model.json',
            ),
            'w',
        ) as json_file:
            json_file.write(model_json)

        self.model.save_weights(
            os.path.join(
                'trained_models',
                'model.h5',
            ),
        )
        print('Saved model to disk')

    def preprocessing(
        self,
        dataset,
        hyperparameters,
    ):
        pass

    def train_model(
        self,
        samples,
        target,
        hyperparameters,
    ):
        self.model.fit(
            x=samples,
            y=target,
            batch_size=hyperparameters['batch_size'],
            epochs=hyperparameters['epochs'],
            validation_split=hyperparameters['validation_split'],
        )

    def evaluating_model(
        self,
        testing_samples,
        targets,
        hyperparameters,
    ):
        raise NotImplemented

    def compile_model(
        self,
        input_shape,
        hyperparameters,
    ):
        raise NotImplemented
