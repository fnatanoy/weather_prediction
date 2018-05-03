import os
import pandas
import training
import preprocessing


def main():
    models_architecture_name = 'double_lstm'
    dataset_config = 'config_1'
    hyperparameters = {
        'window_length': 6,
        'lstm_1_size': 50,
        'lstm_1_dropout': 0.1,
        'dense_1': 50,
        'batch_size': 32,
        'epochs': 1,
        'validation_split': 0.2,
        'test_size': 0.2,
    }

    preprocessor = preprocessing.datasets_structures[dataset_config].Preprocessing(
        from_file=True,
    )
    dataset = preprocessor.get_extracted_dataset(
        data_set_config=dataset_config,
    )
    model = training.models_architectures[models_architecture_name]
    model.run(
        sampels=samels,
        target=target,
        hyperparameters=hyperparameters,
    )


if __name__ == '__main__':
    main()
