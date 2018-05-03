import os
import numpy
import keras
import sklearn
import preprocessing
import plotting_utils


class EvaluateModel:
    model_path = 'trained_model'
    model = ''

    def __init__(
        self,
    ):
        self.preprocessor = preprocessing.preprocessing.Preprocessing()

    def run(
        self,
        dataset,
        labels,
        hyperparameters,
    ):
        pass

    def rems(
        self,
        targets_predictions,
        targets_true,
    ):
        pass

    def preprocessing(
        self,
        dataset,
        hyperparameters,
    ):
        pass

    def load_model(
        self,
    ):
        json_file = open(
            os.path.join(
                'trained_models',
                'model.json',
            ),
            'r',
        )
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = keras.models.model_from_json(loaded_model_json)
        self.model.load_weights(
            os.path.join(
                'trained_models',
                'model.h5',
            )
        )
        self.model.summary()
        print('Loaded model from disk')