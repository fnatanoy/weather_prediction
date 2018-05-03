import os
import pandas
import numpy
import preprocessing
import plotting_utils


def main():
    skip_time_consuming_preprossesing = True
    train_dataset_path = os.path.join(
        'data',
        'train.csv',
    )
    preprocessor = preprocessing.preprocessing.Preprocessing()

    train_dataset = pandas.read_csv(train_dataset_path)

    train_dataset['non_harm'] = preprocessor.get_no_harm_column(
        dataset=train_dataset,
    )
    labels = train_dataset.columns[2:]

    train_dataset['comment_text'] = preprocessor.get_cleaned_comments(
        train_dataset['comment_text'],
    )
    train_dataset['comment_length'] = preprocessor.get_comment_length_column(
        comment_text=train_dataset['comment_text'],
    )
    if not skip_time_consuming_preprossesing:
        train_dataset['comment_spell_checked'] = preprocessor.get_comment_spell_checked_column(
            comment_text=train_dataset['comment_text'],
        )
        train_dataset['comment_spell_checked'] = preprocessor.get_segmented_text_column(
            comment_text=train_dataset['comment_text'],
        )
    tfidf_features, tfidf_mat = preprocessor.get_tfidf(
        comment_text=train_dataset['comment_text']
    )
    preprocessor.initialize_tokenizer(
        comment_text=train_dataset['comment_text'],
        num_words=10000000,
    )

    number_of_samples = train_dataset.shape[0]
    number_of_non_harmful_samples = train_dataset['non_harm'].sum()
    number_of_harmful_samples = number_of_samples - number_of_non_harmful_samples

    print('number of samples = {number_of_samples}'.format(
        number_of_samples=number_of_samples
    ))
    print('number of non harmful comments - {number_of_non_harmful_samples}'.format(
        number_of_non_harmful_samples=number_of_non_harmful_samples,
    ))
    print('number of harmful comments - {number_of_harmful_samples}'.format(
        number_of_harmful_samples=number_of_harmful_samples,
    ))
    print('harmful percentage - {harmful_percentage}%'.format(
        harmful_percentage=round(100 * number_of_harmful_samples / number_of_samples, 1)
    ))
    print('mean comments length  {mean_comment_length}'.format(
        mean_comment_length=round(numpy.mean(train_dataset['comment_length']))
    ))
    print('std comments length  {std_comment_length}'.format(
        std_comment_length=round(numpy.std(train_dataset['comment_length']))
    ))
    print('median comments length  {median_comment_length}'.format(
        median_comment_length=round(numpy.median(train_dataset['comment_length']))
    ))

    target_dist_plotter = plotting_utils.plot_target_dist.TargetDistPlotter()
    target_dist_plotter.plot(
        targets=train_dataset[labels]
    )
    tfidf_plotter = plotting_utils.plot_tfidf.TfidfPlotter()
    tfidf_plotter.plot(
        dataset=train_dataset,
        tfidf_mat=tfidf_mat,
        tfidf_features=tfidf_features,
        labels=labels[:-1],
    )
    comment_length_plotter = plotting_utils.plot_comment_length_hist.CommentLengthPlotter()
    comment_length_plotter.plot(
        comments_length=train_dataset['comment_length'].values
    )
    word_count_obj = preprocessor.get_word_count_obj()
    words_cdf_plotter = plotting_utils.plot_words_cdf.WordsCdfPlotter()
    words_cdf_plotter.plot(
        word_count_obj=word_count_obj,
    )
    print('There are {word_count} words'.format(
        word_count=len((word_count_obj.items()))
    ))


if __name__ == '__main__':
    main()
