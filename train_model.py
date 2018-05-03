import training


def main():
    dataset_config = 'afula_sample'
    models_architecture_name = 'double_lstm'
    hyperparameters = {
        'window_size': 3,
        'lstm_1_size': 3,
        'lstm_1_dropout': 0.1,
        'dense_1': 3,
        'batch_size': 32,
        'epochs': 1,
        'validation_split': 0.2,
        'test_size': 0.2,
    }

    model = training.models_architectures[models_architecture_name]()
    model.run(
        dataset_config=dataset_config,
        hyperparameters=hyperparameters,
    )


if __name__ == '__main__':
    main()
