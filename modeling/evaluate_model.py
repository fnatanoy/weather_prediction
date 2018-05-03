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
        self.load_model()
        samples_pad_sequences = self.preprocessing(
            dataset=dataset,
            hyperparameters=hyperparameters,
        )
        predictions = self.model.predict(
            samples_pad_sequences,
            batch_size=128,
            verbose=1,
        )
        self.calculate_mean_auc(
            targets_predictions=predictions,
            targets_true=dataset[labels],
        )
        roc_plotter = plotting_utils.plot_roc_curve.RocCurvePlotter()
        roc_plotter.plot(
            targets_predictions=predictions,
            targets_true=dataset[labels],
        )

    def calculate_mean_auc(
        self,
        targets_predictions,
        targets_true,
    ):
        auc = [0] * 6
        for label_index, label in enumerate(targets_true.columns):
            label_prediction = [
                targets_prediction[label_index]
                for targets_prediction in targets_predictions
            ]
            auc[label_index] = sklearn.metrics.roc_auc_score(
                y_true=targets_true[label],
                y_score=label_prediction,
            )

        print('Mean auc - {mean_auc}'.format(
            mean_auc=int(numpy.mean(auc) * 1000) / 1000
        ))

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

    def preprocessing(
        self,
        dataset,
        hyperparameters,
    ):
        dataset['comment_text'] = self.preprocessor.get_cleaned_comments(
            dataset['comment_text'],
        )
        self.preprocessor.initialize_tokenizer(
            comment_text=dataset['comment_text'],
            num_words=hyperparameters['max_features'],
        )
        samples_pad_sequences = self.preprocessor.get_pad_sequences(
            dataset['comment_text'],
            maxlen=hyperparameters['maxlen']
        )

        return samples_pad_sequences
