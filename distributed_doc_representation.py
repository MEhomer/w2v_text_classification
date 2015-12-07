'''
    Distributed representation of documents using doc2vec by Quoc Le and Tomas Mikolov.

    Will use that representation as feature vector in classification model.
'''
import os
import inspect

import gensim

import util

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

def train_doc2vec(dataset, size=100, alpha=0.025, window=5, dm=1, hierarchical_softmax=1, negative=0, dm_mean=0, dm_concat=0, dbow_words=0, iterations=1, workers=4, logger_name=__name__):
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

    doc2vec_model = gensim.models.Doc2Vec(size=size, alpha=alpha, window=window, dm=dm, hs=hierarchical_softmax, negative=negative, dm_mean=dm_mean, dbow_words=dbow_words, workers=workers, min_alpha=alpha)

    doc2vec_model.build_vocab(tagged_documents)
    doc2vec_model.train(tagged_documents)

    logger.info('Function={0}, Taggeddocuments={1}, Message="{2}"'.format(
        inspect.currentframe().f_code.co_name,
        len(tagged_documents),
        'Doc2Vec model training finished'
        ))

    return doc2vec_model


def main():
    '''
        Main entry.

        Here goes code for testing the methods and classes of this module.
    '''
    util.setup_logger('DistributedRepresentationLogger', os.path.join('logs', 'distributed_representation.log'))
    dataset = util.read_dataset_threaded(os.path.join('data', 'raw_texts.txt'))

    print train_doc2vec(dataset, logger_name='DistributedRepresentationLogger')

if __name__ == '__main__':
    main()
