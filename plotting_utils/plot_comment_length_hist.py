import matplotlib.pyplot
import numpy
import seaborn


class CommentLengthPlotter:
    def plot(
        self,
        comments_length,
    ):
        seaborn.distplot(
            comments_length,
            kde=False,
        )

        matplotlib.pyplot.ylabel(
            'Frequency',
            fontsize=16,
        )
        matplotlib.pyplot.xlabel(
            '#words per comment',
            fontsize=16,
        )
        matplotlib.pyplot.title(
            'Mean comment length - {mean_comment_length} words, Median - {median_comment_length}'.format(
                mean_comment_length=int(numpy.mean(comments_length)),
                median_comment_length=int(numpy.median(comments_length)),
            ),
            fontsize=16
        )
        matplotlib.pyplot.show()
