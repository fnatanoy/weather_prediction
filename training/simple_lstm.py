import keras

from . import _train_model


class SimpleLSTM(
    _train_model.TrainModel,
):
    models_architecture_name = 'simple_lstm'

    def compile_model(
        self,
        input_shape,
        hyperparameters,
    ):
        inp = keras.layers.Input(
            shape=input_shape,
        )
        x = keras.layers.LSTM(
            hyperparameters['lstm_1_size'],
            return_sequences=False,
            dropout=hyperparameters['lstm_1_dropout'],
            recurrent_dropout=hyperparameters['lstm_1_dropout'],

        )(inp)
        x = keras.layers.Dense(
            hyperparameters['dense_1'],
            activation='relu',
        )(x)
        x = keras.layers.Dense(
            1,
            activation='sigmoid',
        )(x)
        self.model = keras.models.Model(
            inputs=inp,
            outputs=x,
        )
        self.model.compile(
            loss='mean_squared_error',
            optimizer='adam',
        )
        self.model.summary()
