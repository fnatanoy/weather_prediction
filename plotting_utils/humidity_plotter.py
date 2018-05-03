import matplotlib.pyplot


class HumidityPlotter:
    def plot(
        self,
        dataset,
    ):
        print('plotting humidity for Afula')
        humidity_models = [
            humidity_model
            for humidity_model in dataset.columns
            if 'humidity' in humidity_model.lower() and 'persist' not in humidity_model.lower()
        ]
        afula_dataset = dataset[dataset['city'] == 'Afula']

        matplotlib.pyplot.figure()
        ax = matplotlib.pyplot.subplot(111)
        days = range(
            0,
            afula_dataset.shape[0],
        )

        for model in humidity_models:
            ax.plot(
                days,
                afula_dataset[model],
                label=model,
            )

        ax.legend()
        matplotlib.pyplot.show(block=False)
