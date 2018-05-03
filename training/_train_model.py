import os
import numpy
import keras
import sklearn
import preprocessing


class TrainModel:
    models_architecture_name = ''

    def __init__(
        self,
    ):
        self.preprocessor = preprocessing.preprocessing.Preprocessing()

    def run(
        hyperparameters,
    ):
        #preprocessing

        #datasets

        #compile
        self.compile_model(
            hyperparameters=hyperparameters,
        )

        self.train_model(
            training_samples=samples_pad_sequences[training_indexes, :],
            targets=dataset[labels].values[training_indexes, :],
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
        training_samples,
        targets,
        hyperparameters,
    ):
        self.model.fit(
            x=training_samples,
            y=targets,
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
        x,
        y,
        hyperparameters,
    ):
        raise NotImplemented
