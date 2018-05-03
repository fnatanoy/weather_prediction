import matplotlib.pyplot


class TemperaturePlotter:
    def plot(
        self,
        dataset,
    ):
        print('plotting temperature for Afula')
        temperature_models = [
            temperature_model
            for temperature_model in dataset.columns
            if 'temp' in temperature_model.lower() and 'persist' not in temperature_model.lower()
        ]
        afula_dataset = dataset[dataset['city'] == 'Afula']

        matplotlib.pyplot.figure()
        ax = matplotlib.pyplot.subplot(111)
        days = range(
            0,
            afula_dataset.shape[0],
        )

        for model in temperature_models:
            ax.plot(
                days,
                afula_dataset[model],
                label=model,
            )

        ax.legend()
        matplotlib.pyplot.show(block=False)
