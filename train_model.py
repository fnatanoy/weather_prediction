import os
import pandas
import modeling


def main():
    labels = [
        'toxic',
        'severe_toxic',
        'obscene',
        'threat',
        'insult',
        'identity_hate',
    ]

    hyperparameters = {
        'embed_size': 50,
        'max_features': 20000,
        'maxlen': 100,
        'lstm_1_size': 50,
        'lstm_1_dropout': 0.1,
        'dense_1': 50,
        'batch_size': 32,
        'epochs': 1,
        'validation_split': 0.2,
        'test_size': 0.2,
    }

    train_dataset_path = os.path.join(
        'data',
        'train.csv',
    )
    train_dataset = pandas.read_csv(train_dataset_path)

    model_trainer = modeling.train_model.TrainModel()
    model_trainer.run(
        dataset=train_dataset,
        labels=labels,
        hyperparameters=hyperparameters,
    )


if __name__ == '__main__':
    main()
