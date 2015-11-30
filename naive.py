'''
    Here I will use the naive approach for classification of documents. This means that for every
    text, a feature vector will be calculate by adding all the word vectors from its words and
    averaging it. This is weighted average.
'''
import os
import inspect

import gensim
import numpy as np

import util

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

    return tfidf[corpus], dictionary

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

    feature_vec = feature_vec / den

    return feature_vec

def train_word2vec(dataset, size=100, alpha=0.025, window=5, skipgram=1, hierarchical_softmax=1,\
    negative=0, cbow_mean=1, iterations=10, workers=4, logger_name=__name__):
    '''
        Trains a word2vec model using the gensim package.

        Arguments:
            dataset : <list>
                The dataset on which the word2vec model will be trained
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
            cbow_mean : <int>
                If the mean of the context words should be used when using CBOW. 1 means yes and
                0 means no
                Default value is: 1
            iterations : <int>
                Number of iteration through the dataset
                Default value is: 10
            workers : <int>
                Number of workers to be used
                Default value is: 4
        Returns:
            word2vec_mode : <gensim.models.Word2Vec>
                The trained word2vec model
    '''
    logger = util.get_logger(logger_name)
    logger.info('Function={0}, Size={1}, Alpha={2}, Window={3}, Skipgram={4}, HierarchicalSoftmax={5}, NegativeSamplings={6}, CbowMean={7}, Iterations={8}, Workers={9}, Message="{10}"'.format(
        inspect.currentframe().f_code.co_name,
        size, alpha, window, skipgram, hierarchical_softmax,
        negative, cbow_mean, iterations, workers,
        'Word2Vec model training started',
        ))

    sentences = list(sentence_iterator(dataset))

    word2vec_model = gensim.models.Word2Vec(size=size, alpha=alpha, window=window, sg=skipgram,\
        hs=hierarchical_softmax, negative=negative, cbow_mean=cbow_mean, iter=iterations,\
        workers=workers)
    word2vec_model.build_vocab(sentences)
    word2vec_model.train(sentences, total_examples=len(sentences))

    logger.info('Function={0}, Sentences={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        len(sentences),
        'Word2Vec model training finished'
        ))

    return word2vec_model

def main():
    '''
        Main entry.

        Here goes code for testing the methods and classes of this module.
    '''
    util.setup_logger('NaiveLogger', os.path.join('logs', 'naive.log'))

    dataset = util.read_dataset_threaded(os.path.join('data', 'raw_texts.txt'), processes=2)

    word2vec_model = train_word2vec(dataset, iterations=1, logger_name='NaiveLogger')
    tfidf_scores, dictionary = tfidf_model(dataset, logger_name='NaiveLogger')

    tmp = []
    for index, text_tfidf_score in enumerate(tfidf_scores):
        print index
        tmp.append(extract_feature(text_tfidf_score, word2vec_model, dictionary, 100))

if __name__ == '__main__':
    main()
