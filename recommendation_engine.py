from gensim.models import Word2Vec
from vectorizer import TfidfEmbeddingVectorizer as vec
from word2vec import generate_corpus
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_model(path):

    model = Word2Vec.load(path)

    return model

def vectorize(model, corpus):

    tfidf_vec = vec(model)
    tfidf_vec.fit(corpus)
    document_vectors = tfidf_vec.transform(corpus)

    document_vectors = [doc.reshape(1, -1) for doc in document_vectors]

    return tfidf_vec, document_vectors

def vectorize2(model, corpus):

    tfidf_vec = vec(model)
    # tfidf_vec.fit(corpus)

    return tfidf_vec

def get_input_ingredient_embeddings(input_ingredients_list, tfidf_vec):

    input_embedding = tfidf_vec.transform([input_ingredients_list])[0].reshape(1, -1)

    return input_embedding

def get_input_ingredient_embeddings2(input_ingredients_list, tfidf_vec, tfidf_weights):

    tfidf_vec.word_idf_weight = tfidf_weights.item()

    input_embedding = tfidf_vec.transform([input_ingredients_list])[0].reshape(1, -1)

    return input_embedding

def get_similarity_scores(input_embedding, document_vectors):

    cos_sim_scores = map(lambda x: cosine_similarity(input_embedding, x)[0][0], document_vectors)
    scores_list = list(cos_sim_scores)

    return scores_list

def get_similarity_scores2(input_embedding, document_vectors):
    
    document_vectors = np.expand_dims(document_vectors, 1)
    n = np.sum(document_vectors * input_embedding, axis=2)
    d = np.linalg.norm(document_vectors, axis=2) * np.linalg.norm(input_embedding, axis=1)

    return n / d


