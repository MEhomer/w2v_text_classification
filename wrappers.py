'''
    Wrappers for doc2vec and word2vec.
'''
import gensim
import numpy as np

class DocVecs(object):
    """docstring for DocVecs"""
    def __init__(self, doc2vec_models):
        super(DocVecs, self).__init__()
        self.doc2vec_models = doc2vec_models

    def __getitem__(self, key):
        return self.doc2vec_models[key]


class ConcatDoc2vec(object):
    '''
        docstring for ConcatDoc2vec
    '''
    def __init__(self, doc2vec_models, alpha=0.025, min_alpha=0.0001):
        super(ConcatDoc2vec, self).__init__()

        self.alpha = alpha
        self.min_alpha = min_alpha
        self.docvecs = DocVecs(self)
        
        self.doc2vec_models = []
        for doc2vec_model in doc2vec_models:
            if not isinstance(doc2vec_model, gensim.models.doc2vec.Doc2Vec):
                raise ValueError('Incorect value in list')

            doc2vec_model.alpha = self.alpha
            doc2vec_model.min_alpha = self.min_alpha
            self.doc2vec_models.append(doc2vec_model)

    def infer_vector(self, doc_words, alpha=0.1, min_alpha=0.0001, steps=5):
        vector = []
        for doc2vec_model in self.doc2vec_models:
            vector.append(doc2vec_model.infer_vector(doc_words, alpha, min_alpha, steps))

        return np.concatenate(vector)

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None):
        for index in xrange(len(self.doc2vec_models)):
            self.doc2vec_models[index].build_vocab(sentences, keep_raw_vocab, trim_rule)

    def train(self, sentences, total_words=None, word_count=0, total_examples=None, queue_factor=2, report_delay=1.0):
        for index in xrange(len(self.doc2vec_models)):
            self.doc2vec_models[index].alpha = self.alpha
            self.doc2vec_models[index].min_alpha = self.min_alpha

            self.doc2vec_models[index].train(sentences, total_words, word_count,total_examples,queue_factor, report_delay)

    def __getitem__(self, key):
        vector = []
        for doc2vec_model in self.doc2vec_models:
            vector.append(doc2vec_model.docvecs[key])
        
        return np.concatenate(vector)
