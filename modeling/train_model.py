import os
import numpy
import keras
import sklearn
import preprocessing


class TrainModel:
    embedding_file_path = os.path.join(
        'data',
        'glove.6B.50d.txt',
    )
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
        samples_pad_sequences = self.preprocessing(
            dataset=dataset,
            hyperparameters=hyperparameters,
        )
        training_indexes, testing_indexes = self.split_training_testing(
            dataset=dataset,
            hyperparameters=hyperparameters,
        )
        embedding_matrix = self.get_embedding_matrix(
            hyperparameters=hyperparameters,
        )
        self.compile_model(
            embedding_matrix=embedding_matrix,
            hyperparameters=hyperparameters,
        )

        del embedding_matrix

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
        dataset['non_harm'] = self.preprocessor.get_no_harm_column(
            dataset=dataset,
        )
        X = numpy.array(
            range(
                len(dataset['non_harm'])
            )
        )
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X,
            dataset['non_harm'],
            test_size=hyperparameters['test_size'],
            stratify=dataset['non_harm'],
        )

        return (
            X_train,
            X_test,
        )

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

    def get_embedding_matrix(
        self,
        hyperparameters,
    ):
        def get_coefficients(
            word,
            *arr
        ):
            return (
                word,
                numpy.asarray(
                    arr,
                    dtype='float32',
                ),
            )

        embeddings_index = dict(
            get_coefficients(*o.strip().split())
            for o in open(self.embedding_file_path)
        )

        all_embs = numpy.stack(embeddings_index.values())
        emb_mean = all_embs.mean()
        emb_std = all_embs.std()

        word_index = self.preprocessor.get_word_index()
        nb_words = min(
            hyperparameters['max_features'],
            len(word_index),
        )
        embedding_matrix = numpy.random.normal(
            emb_mean,
            emb_std,
            (
                nb_words,
                hyperparameters['embed_size'],
            ),
        )
        for word, i in word_index.items():
            if i >= hyperparameters['max_features']:
                continue

            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def compile_model(
        self,
        embedding_matrix,
        hyperparameters,
    ):
        inp = keras.layers.Input(
            shape=(
                hyperparameters['maxlen'],
            ),
        )
        x = keras.layers.Embedding(
            hyperparameters['max_features'],
            hyperparameters['embed_size'],
            weights=[embedding_matrix]
        )(inp)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(
                hyperparameters['lstm_1_size'],
                return_sequences=True,
                dropout=hyperparameters['lstm_1_dropout'],
                recurrent_dropout=hyperparameters['lstm_1_dropout'],
            )
        )(x)
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
            6,
            activation='sigmoid',
        )(x)
        self.model = keras.models.Model(
            inputs=inp,
            outputs=x,
        )
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )
        self.model.summary()

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
