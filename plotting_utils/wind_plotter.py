import matplotlib.pyplot


class WindPlotter:
    def plot(
        self,
        dataset,
    ):
        print('plotting temperature for Afula')
        wind_models = [
            wind_model
            for wind_model in dataset.columns
            if 'wind' in wind_model.lower() and 'persist' not in wind_model.lower()
        ]
        afula_dataset = dataset[dataset['City'] == 'Afula']

        matplotlib.pyplot.figure()
        ax = matplotlib.pyplot.subplot(111)
        days = range(
            0,
            afula_dataset.shape[0],
        )

        for model in wind_models:
            ax.plot(
                days,
                afula_dataset[model],
                label=model,
            )

        ax.legend()
        matplotlib.pyplot.show(block=False)
