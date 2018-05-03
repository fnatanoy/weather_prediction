import matplotlib.pyplot
import seaborn
import pandas
import numpy


class TfidfPlotter:
    color = seaborn.color_palette()

    def plot(
        self,
        dataset,
        tfidf_mat,
        tfidf_features,
        labels,
        top_n=10,
    ):
        print('calculating top tfidf scores per label')
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.suptitle(
            'TF-IDF top words per label',
            fontsize=20,
        )
        for label_index, label in enumerate(labels):
            print('\tlabel - {label}'.format(
                label=label,
            ))
            comments_label_index = dataset.index[dataset[label] == 1]
            top_tfidf_features = self.get_top_tfidf_features(
                tfidf_mat=tfidf_mat[comments_label_index, :],
                tfidf_features=tfidf_features,
                top_n=top_n,
            )

            matplotlib.pyplot.subplot(
                3,
                2,
                label_index + 1,
            )
            seaborn.barplot(
                top_tfidf_features.feature.iloc[0:top_n],
                top_tfidf_features.tfidf_score.iloc[0:top_n],
                color=self.color[label_index],
                alpha=0.8,
            )
            matplotlib.pyplot.title(
                'label : {label}'.format(
                    label=label,
                ),
                fontsize=15,
            )

            matplotlib.pyplot.xlabel('', fontsize=12)

        matplotlib.pyplot.show()

    def get_top_tfidf_features(
        self,
        tfidf_mat,
        tfidf_features,
        top_n,
    ):
        tfidf_mean_values = numpy.mean(
            tfidf_mat.toarray(),
            axis=0,
        )
        top_n_ids = numpy.argsort(tfidf_mean_values)[::-1][:top_n]
        top_n_features = [
            (
                tfidf_features[index],
                tfidf_mean_values[index],
            )
            for index in top_n_ids
        ]

        top_tfidf_features = pandas.DataFrame(top_n_features)
        top_tfidf_features.columns = [
            'feature',
            'tfidf_score',
        ]

        return top_tfidf_features
