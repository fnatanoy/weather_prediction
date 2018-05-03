import matplotlib.pyplot
import numpy


class WordsCdfPlotter:
    def plot(
        self,
        word_count_obj,
    ):
        word_count = [
            item[1]
            for item in word_count_obj.items()
        ]
        word_count = numpy.asarray(word_count)
        word_count[::-1].sort()
        n_words = numpy.sum(word_count)

        cdf = numpy.cumsum(word_count) / n_words
        matplotlib.pyplot.plot(cdf)
        matplotlib.pyplot.plot(
            3000,
            cdf[3000],
            marker='o',
            markersize=10,
            color='r',
        )

        matplotlib.pyplot.text(
            x=7000,
            y=cdf[3000],
            s=str(round(100 * cdf[3000])) + '%',
            fontsize=14,
        )
        matplotlib.pyplot.xlabel('#Words', fontsize=12)
        matplotlib.pyplot.ylabel('Percentage of accumulated words', fontsize=12)
        matplotlib.pyplot.xlim(-10000, 250000)
        matplotlib.pyplot.ylim(0, 1.1)
        matplotlib.pyplot.show()
