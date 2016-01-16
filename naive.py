'''
    Here I will use the naive approach for classification of documents. This means that for every
    text, a feature vector will be calculate by adding all the word vectors from its words and
    averaging it. This is weighted average.
'''
import os
import copy
import time
import inspect

import gensim
import numpy as np

from sklearn import metrics
from sklearn import linear_model

import util
import cross_validation

def word_list(document):
    '''
        Return a list of all words in one document.

        Arguments:
            document : <tuple>
                A document in format
                    (class_id <str>, link_address <str>, words_in_sentences <list>)
        Returns:
            words : <list>
                A list of all words in this document
    '''
    words = []

    for sentence in document[2]:
        for word in sentence:
            words.append(word)

    return words

def texts_iterator(dataset):
    '''
        Yields a list of words for every text in the dataset.

        Arguments:
            dataset : <list>
                The dataset from where the words for every text will be takne
        Returns:
            words_in_text : <list>
                A list of words in the given text
    '''
    for line in dataset:
        yield word_list(line)

def sentence_iterator(dataset):
    '''
        Yields sentence from the dataset.

        Arguments:
            dataset : <list>
                The dataset from where the sentence to be taken
        Returns:
            line : <list>
                The sentence to be yield
    '''
    for line in dataset:
        for sentence in line[2]:
            yield sentence

def tfidf_model(dataset, logger_name=__name__):
    '''
        Trains a tfidf model from the dataset.

        Arguments:
            dataset : <list>
                The dataset on which the tfidf model will be trained
        Returns:
            tfidf_scores : <list>
                Tfidf scores for every word in every text
            dictionary : <list>
                The word:id mapings for every word in the corpus
            tfidf : <gensim.models.TfidfModel>
                The tfidf trained model
    '''
    logger = util.get_logger(logger_name)

    logger.info('Function={0}, Message="{1}"'.format(
        inspect.currentframe().f_code.co_name,
        'Started calculate tfidf scores'
        ))

    texts = list(texts_iterator(dataset))

    dictionary = gensim.corpora.Dictionary(documents=texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = gensim.models.TfidfModel(corpus)

    logger.info('Function={0}, Texts={2}, VocabSize={2}, Message="{3}"'.format(
        inspect.currentframe().f_code.co_name,
        len(texts), len(dictionary),
        'Finished calculate tfidf scores'
        ))

    return tfidf[corpus], dictionary, tfidf

def extract_feature(text_tfidf_scores, word2vec_model, dictionary, vec_size):
    '''
        Extracts feature vector from the text.

        Arguments:
            text_tfidf_scores : <list>
                The tfidf scores for every word in the text
            word2vec_model : <gensim.model.Word2Vec>
                The trained word2vec model
            dictionary : <gensim.corpora.Dictionary>
                The word:id mapings
            vec_size : <int>
                The size of the vectors
        Returns:
            feature_vec : <numpy.ndarray>
                The feature vector for the text
    '''
    den = 0.0
    feature_vec = np.zeros(vec_size)

    for item in text_tfidf_scores:
        word = dictionary[item[0]]

        if word in word2vec_model.vocab:
            feature_vec += item[1] * word2vec_model[word]
            den += item[1]

    if len(text_tfidf_scores) != 0 and den != 0:
        feature_vec = feature_vec / den
    else:
        feature_vec = np.zeros(vec_size)

    return feature_vec

def make_word2vec(size=100, alpha=0.025, window=5, skipgram=1, hierarchical_softmax=1,\
    negative=0, cbow_mean=1, iterations=1, sample=0, workers=4, logger_name=__name__):
    '''
        Initialize a word2vec model using the gensim package.

        Arguments:
            size : <int>
                Size of the word embedings (vectors) by word2vec
                Default value is: 100
            alpha : <float>
                Starting learning rate of the model
                Default value is: 0.025
            window : <int>
                Size of the context
                Default value is: 5
            skipgram : <int>
                Defines the training algorithm. 1 for Skipgram and 0 for CBOW
                Default value is: 1
            hierarchical_softmax : <int>
                Defines the optimization algorithm. 1 for Hierarchical Softmax, 0 for negative
                Sampling
                Default value is: 1
            negative : <int>
                The number of negative samplings to be used when using Negative Sampling
                Default value is: 0
            cbow_mean : <int>
                If the mean of the context words should be used when using CBOW. 1 means yes and
                0 means no
                Default value is: 1
            iterations : <int>
                Number of iteration through the dataset
                Default value is: 1
            workers : <int>
                Number of workers to be used
                Default value is: 4
        Returns:
            word2vec_model : <gensim.models.Word2Vec>
                The word2vec model
    '''
    logger = util.get_logger(logger_name)
    logger.info('Function={0}, Size={1}, Alpha={2}, Window={3}, Skipgram={4}, HierarchicalSoftmax={5}, NegativeSamplings={6}, CbowMean={7}, Iterations={8}, Sample={9}, Workers={10}, Message="{11}"'.format(
        inspect.currentframe().f_code.co_name,
        size, alpha, window, skipgram, hierarchical_softmax,
        negative, cbow_mean, iterations, workers, sample,
        'Word2Vec model initialized',
        ))

    word2vec_model = gensim.models.Word2Vec(size=size, alpha=alpha, window=window, sg=skipgram,\
        hs=hierarchical_softmax, negative=negative, cbow_mean=cbow_mean, iter=iterations,\
        workers=workers, sample=sample)

    return word2vec_model

def train_word2vec(dataset, word2vec_model, logger_name=__name__):
    '''
        Trains a word2vec model using the gensim package.

        Arguments:
            dataset : <list>
                The dataset on which the word2vec model will be trained
            word2vec_model : <gensim.models.Word2Vec>
                The word2vec model to be trained
        Returns:
            word2vec_model : <gensim.models.Word2Vec>
                The trained word2vec model
    '''
    logger = util.get_logger(logger_name)
    logger.info('Function={0}, Word2Vec={1} Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        word2vec_model,
        'Word2Vec model training started',
        ))

    sentences = list(sentence_iterator(dataset))

    word2vec_model.build_vocab(sentences)
    word2vec_model.train(sentences, total_examples=len(sentences))

    logger.info('Function={0}, Sentences={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        len(sentences),
        'Word2Vec model training finished'
        ))

    return word2vec_model

def evaluate(dataset, model, word2vec_base_model, k_folds=10, shuffle=True, seed=0, logger_name=__name__):
    '''
        Evaluates the models

        Arguments:
            dataset : <list>
                A list of tuples in format
                    (class_id <int>, link_address <str>, words_in_sentences <list>)
                where words_in_sentences is list of list of words
                    [[word <str>, ...], ...]
            model : <sklearn.model>
                A model that will be trained on the training dataset
            word2vec_base_model : <gensim.models.Word2Vec>
                The word2vec base model
            k_folds : <int>
                Number of folds for training/testing
                Default value is 10
            shuffle : <bool>
                Whether to shuffle the data before spliting it
                Default value is True
            seed : <int>
                Sets the seed of the random number generator
                Default value is 0
    '''
    logger = util.get_logger(logger_name)

    logger.info('Function={0}, model={1}, Word2Vec={2}, K-Folds={3}, Shuffle={4}, Seed={5}, Message="{6}"'.format(
        inspect.currentframe().f_code.co_name,
        type(model), word2vec_base_model, k_folds, shuffle, seed,
        'Started evaluating the model',
        ))

    split_data = cross_validation.kfold_items(dataset, k_folds=k_folds, shuffle=shuffle, seed=seed)

    scores = []
    for i in xrange(len(split_data)):
        test_data = split_data[i]

        train_data = []
        for j in xrange(len(split_data)):
            if i != j:
                train_data += split_data[j]

        start_time = time.time()
        word2vec_model = copy.deepcopy(word2vec_base_model)
        word2vec_model = train_word2vec(train_data, word2vec_model, logger_name=logger_name)
        end_time = time.time()
        train_time_word2vec_model = end_time - start_time

        start_time = time.time()
        train_tfidf_scores, dictionary, tfidf = tfidf_model(train_data, logger_name=logger_name)
        end_time = time.time()
        train_time_tfidf_scores = end_time - start_time

        start_time = time.time()
        train_features = []
        for train_text_tfidf_score in train_tfidf_scores:
            train_features.append(extract_feature(train_text_tfidf_score, word2vec_model, dictionary, word2vec_model.vector_size))
        end_time = time.time()
        train_time_feature_extraction = end_time - start_time

        train_classes = []
        for index in xrange(len(train_data)):
            train_classes.append(train_data[index][0])

        start_time = time.time()
        current_model = copy.deepcopy(model)
        current_model.fit(train_features, train_classes)
        end_time = time.time()
        train_time_classification_model = end_time - start_time

        start_time = time.time()
        test_texts = list(texts_iterator(test_data))
        test_corpus = [dictionary.doc2bow(text) for text in test_texts]
        test_tfidf_scores = tfidf[test_corpus]
        test_features = []
        for test_text_tfidf_score in test_tfidf_scores:
            test_features.append(extract_feature(test_text_tfidf_score, word2vec_model, dictionary, word2vec_model.vector_size))
        end_time = time.time()
        test_time_feature_extraction = end_time - start_time

        test_classes = []
        for index in xrange(len(test_data)):
            test_classes.append(test_data[index][0])

        start_time = time.time()
        prediction_classes = current_model.predict(test_features)
        end_time = time.time()
        test_time_prediction = end_time - start_time

        score = current_model.score(test_features, test_classes)
        scores.append(score)

        target_names = [util.INVERSE_CLASS_MAP[key] for key in range(len(util.INVERSE_CLASS_MAP))]
        confusion_matrix = metrics.confusion_matrix(test_classes, prediction_classes)
        classification_report = metrics.classification_report(test_classes, prediction_classes, target_names=target_names)

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            train_time_word2vec_model,
            'Train time of the word2vec model'
            ))

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            train_time_tfidf_scores,
            'Train time of the tfidf model'
            ))

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            train_time_feature_extraction,
            'Required time for feature extraction of the train set'
            ))

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            train_time_classification_model,
            'Train time of the classification model'
            ))

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            test_time_feature_extraction,
            'Required time for feature extraction of the test test'
            ))

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            test_time_prediction,
            'Prediction time of the classification model'
            ))

        logger.info('Function={0}, Message="{1}", ConfusionMatrix=\n{2}'.format(
            inspect.currentframe().f_code.co_name,
            'Confusion matrix of the classification model',
            confusion_matrix
            ))

        logger.info('Function={0}, Message="{1}", ClassificationReport=\n{2}'.format(
            inspect.currentframe().f_code.co_name,
            'Classification report of the classification model',
            classification_report
            ))

    logger.info('Function={0}, Score={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        sum(scores) / float(len(scores)),
        'Mean score of the model',
        ))

def main():
    '''
        Main entry.

        Here goes code for testing the methods and classes of this module.
    '''
    logger_name = 'NaiveLogger'
    util.setup_logger(logger_name, os.path.join('logs', 'naive.log'))

    dataset = util.read_dataset_threaded(os.path.join('data', 'raw_texts.txt'), processes=2,\
        logger_name=logger_name)

    word2vec_base_model = make_word2vec(iterations=1, size=250, logger_name=logger_name)
    model = linear_model.LogisticRegression()

    evaluate(dataset, model, word2vec_base_model, k_folds=6, logger_name=logger_name)

if __name__ == '__main__':
    main()
