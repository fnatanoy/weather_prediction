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



if __name__ == '__main__':
    main()
