'''
    Distributed representation of documents using doc2vec by Quoc Le and Tomas Mikolov.

    Will use that representation as feature vector in classification model.
'''
import os
import copy
import time
import random
import inspect

import gensim

from sklearn import metrics
from sklearn import linear_model

import util
import cross_validation

def word_list(document):
    '''
        Returns a list of all words in one documnet

        Arguments:
            document : <tuple>
                A sample document from the dataset.
                The tuple is in format
                    (class_id <int>, link_address <str>, words_in_sentences <list>)
                where words_in_sentences is list of list of unicode strings
                    [[word <unicode>, ...], ...]
        Returns:
            words : <list>
                A list of all words in the document
    '''
    words = []
    for sentence in document[2]:
        for word in sentence:
            words.append(word)

    return words

def to_tagged_document(document):
    '''
        Converts the document to gensim.models.doc2vec.TaggedDocument

        Arguments:
            document : <tuple>
                A sample document from the dataset.
                The tuple is in format
                    (class_id <int>, link_address <str>, words_in_sentences <list>)
                where words_in_sentences is list of list of unicode strings
                    [[word <unicode>, ...], ...]
        Returns:
            tagged_document : <gensim.models.doc2vec.TaggedDocument>
                The TaggedDocument for this document where the tag is this texts link_address
    '''
    _, link_address, _ = document
    words = word_list(document)

    tagged_document = gensim.models.doc2vec.TaggedDocument(words=words, tags=[link_address])

    return tagged_document

def extract_features(document, doc2vec_model, trained=True):
    '''
        Returns the doc vector for this document from the doc2vec model

        Arguments:
            document : <tuple>
                A sample document from the dataset.
                The tuple is in format
                    (class_id <int>, link_address <str>, words_in_sentences <list>)
                where words_in_sentences is list of list of unicode strings
                    [[word <unicode>, ...], ...]
            doc2vec_model : <gensim.models.doc2vec.Doc2Vec>
                A trained doc2vec model
            trained : <bool>
                Signals if the doc2vec model was trained on this document or not
        Returns:
            document_vector : <numpy.ndarray>
                The vector for this document
    '''
    if trained is True:
        return doc2vec_model.docvecs[document[1]]
    else:
        return doc2vec_model.infer_vector(word_list(document))

def train_doc2vec(dataset, size=100, alpha=0.025, min_alpha=0.001, window=5, dm=1, hierarchical_softmax=1, negative=0, dm_mean=0, dm_concat=0, dbow_words=0, iterations=1, workers=4, logger_name=__name__):
    '''
        Trains a doc2vec model using the gensim package.

        Arguments:
            dataset : <list>
                The dataset on which the doc2vec model will be trained
            size : <int>
                Size of the feature vectors by doc2vec
                Default value is: 100
            alpha : <float>
                Starting learning rate of the model
                Default value is: 0.025
            min_alpha : <float>
                Minimum learning rate of the model
                Default value is: 0.001
            window : <int>
                Size of the context
                Default value is: 5
            dm : <int>
                Defines the training algorithm. 1 for PV-DM and 0 for PV-DBOW
                Default value is: 1
            hierarchical_softmax : <int>
                Defines the optimization algorithm. 1 for Hierarchical Softmax, 0 for Negative
                Sampling
                Default value is: 1
            negative : <int>
                The number of negative samplings to be used when using Negative Sampling
            db_mean : <int>
                If the mean of the context words should be used. 1 means yes and 0 means no
                Default value is: 0
            db_concat : <int>
                Using concatenation of context words rather than sum/average.
                Default value is: 0
            dbow_words : <int>
                If it should learn word-vectors.
                Default value is: 0
            iterations : <int>
                Number of iteration through the dataset
                Default value is: 1
            workers : <int>
                Number of workers to be used
                Default value is: 4
        Returns:
            doc2vec_model : <gensim.models.Doc2Vec>
                The trained doc2vec model
    '''
    logger = util.get_logger(logger_name)
    logger.info('Function={0}, Size={1}, Alpha={2}, Window={3}, PV-DM/PV-DBOW={4}, HierarchicalSoftmax={5}, NegativeSamplings={6}, DMMean={7}, DMConcat={8}, DBOWWords={9}, Iterations={10}, Workers={11}, Message="{12}"'.format(
        inspect.currentframe().f_code.co_name,
        size, alpha, window, dm, hierarchical_softmax, negative, dm_mean, dm_concat, dbow_words,
        iterations, workers,
        "Doc2Vec model training started"
        ))

    tagged_documents = [to_tagged_document(document) for document in dataset]

    doc2vec_model = gensim.models.Doc2Vec(size=size, alpha=alpha, window=window, dm=dm, hs=hierarchical_softmax, negative=negative, dm_mean=dm_mean, dbow_words=dbow_words, workers=workers)
    doc2vec_model.build_vocab(tagged_documents)

    delta_alpha = float(alpha - min_alpha) / float(iterations)

    for iteration in range(iterations):
        doc2vec_model.alpha = alpha
        doc2vec_model.min_alpha = alpha

        random.shuffle(tagged_documents)
        doc2vec_model.train(tagged_documents)

        alpha -= delta_alpha

    logger.info('Function={0}, Taggeddocuments={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        len(tagged_documents),
        'Doc2Vec model training finished'
        ))

    return doc2vec_model

def evaluate(dataset, model, k_folds=10, shuffle=True, seed=0, logger_name=__name__):
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
            k_folds : <int>
                Number of folds for training/testing
                Default value is 10
            shuffle : <bool>
                Whether to shuffle the data before spliting it
                Default value is True
            seed : <int>
                Sets the seed of the random number generator
    '''
    logger = util.get_logger(logger_name)

    logger.info('Function={0}, model={1}, k_folds={2}, shuffle={3}, seed={4}, Message="{5}"'.format(
        inspect.currentframe().f_code.co_name,
        type(model), k_folds, shuffle, seed,
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
        doc2vec_model = train_doc2vec(train_data, logger_name=logger_name, window=10, size=300, workers=8, iterations=25)
        end_time = time.time()
        train_time_doc2vec_model = end_time - start_time

        start_time = time.time()
        train_features = []
        train_classes = []
        for index in xrange(len(train_data)):
            train_features.append(extract_features(train_data[index], doc2vec_model))
            train_classes.append(train_data[index][0])

        end_time = time.time()
        train_time_feature_extraction = end_time - start_time

        start_time = time.time()
        test_features = []
        test_classes = []
        for index in xrange(len(test_data)):
            test_features.append(extract_features(test_data[index], doc2vec_model, trained=False))
            test_classes.append(test_data[index][0])
        end_time = time.time()
        test_time_feature_extraction = end_time - start_time

        start_time = time.time()
        current_model = copy.deepcopy(model)
        current_model.fit(train_features, train_classes)
        end_time = time.time()
        train_time_classification_model = end_time - start_time

        start_time = time.time()
        predicted_classes = current_model.predict(test_features)
        end_time = time.time()
        test_time_prediction = end_time - start_time

        score = current_model.score(test_features, test_classes)
        scores.append(score)

        target_names = [util.INVERSE_CLASS_MAP[key] for key in range(len(util.INVERSE_CLASS_MAP))]

        confusion_matrix = metrics.confusion_matrix(test_classes, predicted_classes)
        classification_report = metrics.classification_report(test_classes, predicted_classes, target_names=target_names)

        logger.info('Function={0}, Time={1}, Message="{2}"'.format(
            inspect.currentframe().f_code.co_name,
            train_time_doc2vec_model,
            'Train time of the doc2vec model'
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
            'Required time for feature extraction of the test set'
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
    util.setup_logger('DistributedRepresentationLogger', os.path.join('logs', 'distributed_representation.log'))

    dataset = util.read_dataset_threaded(os.path.join('data', 'raw_texts.txt'), processes=2)

    model = linear_model.LogisticRegression()

    evaluate(dataset, model, k_folds=6, logger_name='DistributedRepresentationLogger')

if __name__ == '__main__':
    main()
