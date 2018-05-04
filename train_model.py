import training


def main():
    dataset_config = 'afula_sample'
    models_architecture_name = 'simple_lstm_cnn'
    hyperparameters = {
        'window_size': 7,
        'lstm_1_size': 7,
        'lstm_1_dropout': 0.1,
        'dense_1': 7,
        'cnn_number_of_filter': 3,
        'cnn_kernel_size': 3,
        'batch_size': 4,
        'epochs': 20,
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
