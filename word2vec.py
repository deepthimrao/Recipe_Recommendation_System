from gensim.models import Word2Vec
import pandas as pd
import numpy as np

def generate_corpus(data):
    """
    """
    sorted_corpus = []
    for doc in data['NER']:
        doc=doc.strip('][')
        doc=doc.replace('"','')
        doc=doc.split(',')
        doc=[x.strip() for x in doc]
        doc.sort()
        sorted_corpus.append(doc)

    return sorted_corpus

def get_window_length(corpus):
    """
    """
    all_lengths = [len(document) for document in corpus]
    avg_length = float(sum(all_lengths)) / len(all_lengths)

    return round(avg_length)

def train(corpus, sg=0, workers=8, min_count=1, vector_size=100):
    """
    """
    model = Word2Vec(corpus, sg=sg, workers=workers, window=get_window_length(corpus), min_count=min_count, vector_size=vector_size)

    return model

def save_model(model):
    """
    """
    model.save('models/word_2_vec_model_cbow.bin')
    print("Model succesfully saved")

# if __name__ == "__main__":
    
#     data = pd.read_csv('../Datasets/dataset/full_dataset.csv')
    
#     # data = data.iloc[0:231142,]

#     # corpus = generate_corpus(data)

#     corpus = np.load("corpus_data.npy", allow_pickle=True)

#     print(f"Length of corpus: {len(corpus)}")

#     model = train(corpus=corpus, sg=0, workers=8, min_count=1, vector_size=100)

#     save_model(model)
    