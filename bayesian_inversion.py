'''
    Here I will use classifier by inversion of distributed representation of documents via the
    Bayes Rule which was proposed in
        Document Classification by Inversion of Distributed Language Representations by Matt Taddy
'''
import os
import copy
import time
import inspect

import gensim
import pandas as pd
import numpy as np

from sklearn import metrics

import util
import cross_validation

def split_dataset(dataset):
    '''
        Splits the dataset by category

        Arguments:
            dataset : <list>
                The dataset to be splited
        Returns:
            dataset_by_category : <dict>
                The splited dataset by category
                Keys are the categories and values are list of the lines
    '''
    dataset_by_category = {}

    for line in dataset:
        category = line[0]

        if category in dataset_by_category:
            dataset_by_category[category].append(line)
        else:
            dataset_by_category[category] = [line]

    return dataset_by_category

def texts_iterator(dataset, class_id):
    '''
        Gets line from the dataset where the line class_id is equal to class_id

        Arguments:
            dataset : <list>
                The dataset from where the line needs to be taken
            class_id : <int>
                The class_id which should be extracted
        Yields:
            line : <list>
                The line where the class_id is equal to class_id
    '''
    for line in dataset:
        if line[0] in class_id:
            yield line

def sentence_iterator(dataset, class_id):
    '''
        Yields sentences from the texts of the given class_id

        Arguments:
            dataset : <list>
                The dataset from where the sentences to be taken
            class_id : <int>
                The class_id which should be extracted
        Yields:
            sentence : <list>
                A list of words
    '''
    for line in dataset:
        if line[0] in class_id:
            for sentence in line[2]:
                yield sentence

def train_word2vec(dataset, num_categories, size=100, alpha=0.025, window=5, skipgram=1, \
    hierarchical_softmax=1, negative=0, cbow_mean=1, iterations=10, workers=4,\
     logger_name=__name__):
    '''
        Trains a word2vec model for every category in the dataset defined by num_categories.

        Arguments:
            dataset : <list>
                The dataset on which the word2vec models will be trained
            num_categories : <int>
                The number of categories in dataset, also the number of word2vec models
                which will be trained
            size : <int>
                Size of the word embedings (vectors) by word2vec
                Default value is: 100
            alpha : <float>
                Starting learning rate for every word2vec model
                Default value is: 0.025
            window : <int>
                Size of the context
                Default value is: 5
            sg : <int>
                Defines the training algorithm. 1 for Skipgram and 0 for CBOW
                Default value is: 1
            hs : <int>
                Defines the optimization algorithm. 1 for Hierarchical Softmax and 0 for Negative
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
                Default value is: 10
            workers : <int>
                Number of workers to be used
                Default value is: 4
        Returns:
            word2vec_models : <list>
                List of trained word2vec models

    '''
    logger = util.get_logger(logger_name)

    logger.info('Function={0}, NumCategories={1}, Size={2}, Alpha={3}, Window={4}, Skipgram={5}, HierarchicalSoftmax={6}, NegativeSamplers={7}, CbowMean={8}, Iterations={9}, Workers={10}, Message="{11}"'.format(
            inspect.currentframe().f_code.co_name,
            num_categories, size, alpha, window, skipgram, 
            hierarchical_softmax, negative, cbow_mean, iterations, workers,
            'Word2vec model training started'
            ))

    basemodel = gensim.models.Word2Vec(size=size, alpha=alpha, window=window, sg=skipgram, hs=hierarchical_softmax, negative=negative, cbow_mean=cbow_mean, iter=iterations, workers=workers)
    basemodel.build_vocab(sentence_iterator(dataset, range(num_categories)))

    models = [copy.deepcopy(basemodel) for i in range(num_categories)]

    for class_id in range(num_categories):
        slist = list(sentence_iterator(dataset, class_id=[class_id]))
        models[class_id].train(slist, total_examples=len(slist))
        
        logger.info('Function={0}, Class={1}, Sentences={2}, Message="{3}"'.format(
            inspect.currentframe().f_code.co_name,
            util.INVERSE_CLASS_MAP[class_id], len(slist),
            'Number of sentences in class'
            ))

    return models

def docprob(doc, models):
    '''
        Calculates the probability of the document

        Arguments:
            doc : <list>
                The document to which the probability shoudl be calculated
                It is list of list (sentences) of words
            models : <list>
                A list of the word2vec models
        Returns:
            prob : <list>
                The probability of every class
    '''
    log_probability_sent = np.array([model.score(doc, len(doc)) for model in models])
    prob_sent = np.exp(log_probability_sent - log_probability_sent.max(axis=0))
    prob = (prob_sent / prob_sent.sum(axis=0)).transpose()
    prob = prob.sum(axis=0) / prob.shape[0]

    return prob.tolist()

def docsprob(docs, models, logger_name=__name__):
    '''
        Calculates the probability of the documents

        Arguments:
            doc : <list>
                The documents to which the probability should be calculated
                It is list of tuples in the next format
                (class_id <str>, link_address <str>, words_in_sentences <list)
                where words_in_sentences is list of list of words
                    [[word <str>, ...], ...]
            models : <list>
                A list of the word2vec models
        Returns:
            prob : <list>
                The probability of every document for every class
    '''
    logger = util.get_logger(logger_name)

    logger.info('Function={0}, #Docs={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        len(docs),
        'Calculating document probability for #Docs'
        ))

    sentlist = [sentence for doc in docs for sentence in doc[2]]

    log_probability_sent = np.array([model.score(sentlist, len(sentlist)) for model in models])
    prob_sent = np.exp(log_probability_sent - log_probability_sent.max(axis=0))

    prob = pd.DataFrame((prob_sent/prob_sent.sum(axis=0)).transpose())
    prob["doc"] = [i for i, doc in enumerate(docs) for s in doc[2]]
    prob = prob.groupby("doc").mean()

    return np.array(prob).tolist()

def evaluate(dataset, k_folds=10, shuffle=True, seed=0, logger_name=__name__):
    '''
        Evaluate the models

        Arguments:
            dataset : <list>
                A list of tuples in format
                    (class_id <str>, link_address <str>, words_in_sentences <list>)
                where words_in_sentences is list of list of words
                    [[word <str>, ...], ...]
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

    split_data = cross_validation.kfold_items(dataset, k_folds=k_folds, shuffle=shuffle, seed=seed)

    scores = []
    for i in xrange(len(split_data)):
        test_data = split_data[i]

        train_data = []
        for j in xrange(len(split_data)):
            if i != j:
                train_data += split_data[j]

        start_time = time.time()
        models = train_word2vec(train_data, len(util.CLASS_MAP), iterations=1, size=300,\
            logger_name=logger_name)
        end_time = time.time()
        train_time = end_time - start_time

        start_time = time.time()
        probs = docsprob(test_data, models, logger_name=logger_name)
        end_time = time.time()
        prediction_time = end_time - start_time

        predictions = []
        for i in xrange(len(probs)):
            predictions.append(probs[i].index(max(probs[i])))

        trues = []
        for i in xrange(len(test_data)):
            trues.append(test_data[i][0])

        correct_count = 0
        for j in xrange(len(predictions)):
            if predictions[j] == test_data[j][0]:
                correct_count += 1

        target_names = [util.INVERSE_CLASS_MAP[key] for key in range(len(util.INVERSE_CLASS_MAP))]
        confusion_matrix = metrics.confusion_matrix(trues, predictions)
        classification_report = metrics.classification_report(trues, predictions, target_names=target_names)

        score = float(correct_count) / float(len(test_data))
        scores.append(score)

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            train_time,
            'Training time of the model'
            ))

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            prediction_time,
            'Prediction time of the model'
            ))

        logger.info('Function={0}, Message="{1}", ConfusionMatrix=\n{2}'.format(
            inspect.currentframe().f_code.co_name,
            'Confusion matrix of the moddel',
            confusion_matrix
            ))

        logger.info('Function={0}, Message="{1}", ClassificationReport=\n{2}'.format(
            inspect.currentframe().f_code.co_name,
            'Classification report of the moddel',
            classification_report
            ))

    logger.info('Function={0}, Score={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        sum(scores) / float(len(scores)),
        'Mean score of the model'
        ))

def main():
    '''
        Main entry.

        Here goes code for testing the methods and classes of this module.
    '''
    util.setup_logger('BayesianLogger', os.path.join('logs', 'bayesian_inversion.log'))

    dataset = util.read_dataset_threaded(os.path.join('data', 'raw_texts.txt'), processes=10)

    evaluate(dataset, k_folds=6, logger_name='BayesianLogger')

if __name__ == '__main__':
    main()
