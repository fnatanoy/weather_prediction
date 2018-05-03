import matplotlib.pyplot
import numpy


class TargetDistPlotter:
    def plot(
        self,
        targets,
    ):
        targets = targets.sum()
        labels = numpy.arange(0, len(targets)) + 1
        matplotlib.pyplot.figure()
        bars = matplotlib.pyplot.bar(
            labels,
            targets.values,
            align='center',
            alpha=0.6,
        )

        matplotlib.pyplot.xticks(
            labels,
            targets.keys(),
            fontsize=15,
        )
        ax = matplotlib.pyplot.gca()
        ax.set_title(
            'Number of samples from each label',
            fontsize=20,
        )
        matplotlib.pyplot.tick_params(
            top='off',
            bottom='off',
            left='off',
            right='off',
            labelleft='off',
            labelbottom='on',
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                x=bar.get_x() + bar.get_width() / 2,
                y=height + 2000,
                s=str(int(height)),
                ha='center',
                color='k',
                fontsize=15,
            )
        for spine in matplotlib.pyplot.gca().spines.values():
            spine.set_visible(False)

        x = matplotlib.pyplot.gca().xaxis
        for item in x.get_ticklabels():
            item.set_rotation(45)

        matplotlib.pyplot.show()
