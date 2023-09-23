import recommendation_engine as rec_eng
from vectorizer import TfidfEmbeddingVectorizer as vec
import word2vec as wv
import pandas as pd
import numpy as np
import dill
import time

WORD2VEC_PATH = "models/word_2_vec_model_cbow.bin"
CORPUS_PATH = "models/corpus_data.npy"
TFIDF_MODEL_PATH = "models/tfidf_wts.pkl"
VECTORS_PATH = "models/all_ingredient_vectors.npy"

w2v_model=None
corpus=None
tfidf_model=None
doc_vectors=None

def load_models():
    
    global w2v_model, corpus, tfidf_model, doc_vectors

    # w2v_model = rec_eng.load_model(WORD2VEC_PATH)
    # w2v_model.init_sims(replace=True)

    # corpus = np.load(CORPUS_PATH,allow_pickle=True)

    f=open(TFIDF_MODEL_PATH,"rb")
    tfidf_model = dill.loads(f.read())

    doc_vectors = np.load(VECTORS_PATH,allow_pickle=True)
    doc_vectors = doc_vectors.reshape(doc_vectors.shape[0],doc_vectors.shape[2])


def generate_recommendations(N, scores, data):

    # top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    
    top_n_indices = scores.argsort()[::-1][:N]
    recommendation = pd.DataFrame(columns=["title", "ingredients", "recipe", "source"])

    count = 0
    for index in top_n_indices:
        recommendation.at[count, "title"] = data["title"][index]
        recommendation.at[count, "ingredients"] = data["ingredients"][index]
        recommendation.at[count, "recipe"] = data["directions"][index]
        recommendation.at[count, "source"] = data["link"][index]
        count += 1
    
    return recommendation

def process_input_ingredients(input_ingredients):

    input_ingredients_list = input_ingredients.split(",")
    input_ingredients_list = [l.strip() for l in input_ingredients_list]

    return input_ingredients_list

def get_input_scores(input_ingredients_list, tfidf_model, doc_vectors):

    inp_emb = rec_eng.get_input_ingredient_embeddings(input_ingredients_list, tfidf_model)
    scores = rec_eng.get_similarity_scores2(inp_emb, doc_vectors)
    scores=scores.reshape(scores.shape[0])

    return scores

if __name__ == "__main__":
    # input_ingredients = "chicken thigh, onion, rice noodle, seaweed nori sheet, sesame, shallot, soy, spinach, star, tofu"

    start=time.time()
    # loading of models needs to be done only once in the beginning
    load_models()

    mid = time.time()
    #processing of input - done everytime
    input_ingredients = "cream cheese, Worcestershire sauce, hot sauce, Cheddar cheese"    
    input_ingredients_list = process_input_ingredients(input_ingredients)
    scores = get_input_scores(input_ingredients_list, tfidf_model, doc_vectors)
    end = time.time()

    print("Time taken to load models = {} sec".format(mid-start))
    print("Time taken to process input and get scores = {} sec".format(end-mid))

    #loading of dataset - needs to be done only once in beginning
    data = pd.read_csv('../Datasets/dataset/full_dataset.csv')

    #generation of recommendations
    r = generate_recommendations(5, scores, data)

    print(r)
