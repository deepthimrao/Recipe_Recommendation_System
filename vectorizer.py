from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np


class TfidfEmbeddingVectorizer(object):

    def __init__(self, word2vec_model):

        self.model = word2vec_model
        self.word_idf_weight = None
        self.vector_len = word2vec_model.wv.vector_size

    def fit(self, corpus): 

        text_document_list = []
        for doc in corpus:
            text_document_list.append(" ".join(doc))
        
        tfidf = TfidfVectorizer()
        tfidf.fit(text_document_list)
        max_idf = max(tfidf.idf_)  
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )

        
        

    def transform(self, corpus): 
        doc_word_vector = self.doc_average_list(corpus)

        return doc_word_vector

    def aggregate_embeddings(self, document):
        mean_list = []
        for word in document:
            if word in self.model.wv.index_to_key:
                mean_list.append(self.model.wv.get_vector(word) * self.word_idf_weight[word]) 

        if not mean_list:  
            return np.zeros(self.vector_len)
        else:
            mean = np.array(mean_list).mean(axis=0)
            return mean

    def doc_average_list(self, docs):
        return np.vstack([self.aggregate_embeddings(doc) for doc in docs])