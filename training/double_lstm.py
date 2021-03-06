import keras

from . import _train_model


class DoubleLSTM(
    _train_model.TrainModel,
):
    models_architecture_name = 'double_lstm'

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
            return_sequences=True,
            dropout=hyperparameters['lstm_1_dropout'],
            recurrent_dropout=hyperparameters['lstm_1_dropout'],

        )(inp)
        x = keras.layers.Dropout(
            0.1
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(
            hyperparameters['lstm_1_size'],
            return_sequences=False,
        )(x)
        x = keras.layers.Dropout(0.1)(x)
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
