import os
import pandas
import numpy
import preprocessing
import plotting_utils


def main():
    file_path = os.path.join(
        'data',
        'all_data_without_minTemp.pkl',
    )
    dataset = pandas.read_pickle(
        path=file_path,
    )
    plotting_utils.temperature_plotter.TemperaturePlotter().plot(
        dataset=dataset,
    )
    plotting_utils.humidity_plotter.HumidityPlotter().plot(
        dataset=dataset,
    )
    plotting_utils.wind_plotter.WindPlotter().plot(
        dataset=dataset,
    )

    input('boo')

if __name__ == '__main__':
    main()
