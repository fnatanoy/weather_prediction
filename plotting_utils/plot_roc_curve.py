import matplotlib.pyplot
import sklearn.metrics


class RocCurvePlotter:
    def plot(
        self,
        targets_predictions,
        targets_true,
    ):
        print('plotting roc curves per label')
        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.suptitle(
            'ROC Curves',
            fontsize=20,
        )
        for label_index, label in enumerate(targets_true.columns):
            predicted_labels_probabilities = [
                targets_prediction[label_index]
                for targets_prediction in targets_predictions
            ]
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                y_true=targets_true[label],
                y_score=predicted_labels_probabilities,
            )
            roc_auc_lr = sklearn.metrics.auc(
                x=fpr,
                y=tpr,
            )
            matplotlib.pyplot.subplot(
                3,
                2,
                label_index + 1,
            )
            matplotlib.pyplot.plot(
                fpr,
                tpr,
                lw=3
            )
            matplotlib.pyplot.xlim(
                [
                    -0.01,
                    1.00,
                ],
            )
            matplotlib.pyplot.ylim(
                [
                    -0.01,
                    1.01,
                ],
            )
            if label_index + 1 in [5, 6]:
                matplotlib.pyplot.xlabel(
                    'False Positive Rate',
                    fontsize=16,
                )
            if label_index + 1 in [1, 3, 5]:
                matplotlib.pyplot.ylabel(
                    'True Positive Rate',
                    fontsize=16,
                )
            matplotlib.pyplot.title(
                'label - {label}, AUC = {roc_auc_lr:0.2f})'.format(
                    label=label,
                    roc_auc_lr=roc_auc_lr
                ),
                fontsize=16
            )

            matplotlib.pyplot.plot(
                [
                    0,
                    1,
                ],
                [
                    0,
                    1,
                ],
                color='navy',
                lw=3,
                linestyle='--',
            )

        matplotlib.pyplot.show()
