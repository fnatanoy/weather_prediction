import wordsegment
import nltk.stem.wordnet
import sklearn.feature_extraction.text
import keras.preprocessing.text

import re

from . import spelchek


class Preprocessing:
    tokenizer = None

    def __init__(
        self,
    ):
        self.spelchek = spelchek.spelchek

    def get_cleaned_comments(
        self,
        comment_text,
    ):
        def cleaner(
            text,
        ):
            filters_regex = r'(?!\')(?:\W|_)'
            clean_text = re.sub(
                pattern=filters_regex,
                repl=' ',
                string=text.lower(),
            )
            clean_text = re.sub(
                pattern=r'\s+',
                repl=' ',
                string=clean_text,
            )

            return clean_text

        clean_comments = comment_text.apply(
            lambda x: cleaner(x)
        )

        return clean_comments

    def get_no_harm_column(
        self,
        dataset,
    ):
        harm_indicator = dataset.iloc[:, 2:].sum(
            axis=1,
        )
        no_harm_column = harm_indicator == 0

        return no_harm_column

    def get_comment_length_column(
        self,
        comment_text,
    ):
        comment_length = comment_text.apply(
            lambda x: len(x.split())
        )

        return comment_length

    def get_comment_spell_checked_column(
        self,
        comment_text,
    ):
        def checker(
            text,
        ):
            spell_checked_text_words = [
                self.spelchek.correct(
                    word=word,
                )
                for word in text.split()
            ]

            return ' '.join(spell_checked_text_words)

        comment_spell_checked_column = comment_text.apply(
            lambda x: checker(x)
        )

        return comment_spell_checked_column

    def get_segmented_text_column(
        self,
        comment_text,
    ):
        wordsegment.load()

        def segment_text(
            text,
        ):
            segmented_words = [
                wordsegment.segment(
                    text=word,
                )
                for word in text.split()
            ]
            seperated_words = [
                word
                for segment_text in segmented_words
                for word in segment_text
            ]

            segmented_text = ' '.join(seperated_words)

            return segmented_text

        segmented_text_column = comment_text.apply(
            lambda x: segment_text(x)
        )

        return segmented_text_column

    def get_stemmed_text_column(
        self,
        comment_text,
    ):
        raise NotImplemented

    def get_tfidf(
        self,
        comment_text,
    ):
        tfidf_obj = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=10000,
            strip_accents='unicode',
            analyzer='word',
            stop_words='english',
        )
        tfidf_obj.fit(comment_text)

        features = tfidf_obj.get_feature_names()
        tfidf_mat = tfidf_obj.transform(comment_text)

        return (
            features,
            tfidf_mat,
        )

    def initialize_tokenizer(
        self,
        comment_text,
        num_words,
    ):
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=num_words
        )
        self.tokenizer.fit_on_texts(
            texts=list(comment_text),
        )

    def get_word_count_obj(
        self,
    ):
        return self.tokenizer.word_counts

    def get_pad_sequences(
        self,
        comment_text,
        maxlen,
    ):
        list_tokenized_train = self.tokenizer.texts_to_sequences(list(comment_text))

        return keras.preprocessing.sequence.pad_sequences(
            sequences=list_tokenized_train,
            maxlen=maxlen,
        )

    def get_word_index(
        self,
    ):
        return self.tokenizer.word_index
