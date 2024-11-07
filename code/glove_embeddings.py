import numpy as np
from named_entity_recognition  import NER
import pickle

class GLOVE_EMBEDDINGS:
    def __init__(self,GLOVE_FILE_PATH):
        self.dynamic_emb_dict = {} 
        self.GLOVE_FILE_PATH = GLOVE_FILE_PATH
        self.glove_embeddings_index = self.load_glove_embeddings()
        
    def create_dynamic_dict_file(self):
        with open("dynamic_dict.pkl", 'wb') as f:
            pickle.dump(self.dynamic_emb_dict,f)

    def load_glove_embeddings(self):
        embeddings_index = {}
        with open(self.GLOVE_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings_index[word] = vector
        return embeddings_index


    def get_embedding(self,word):
        # Return GloVe embedding if available, else generate/store a dynamic one
        word=word.lower()
        if word in self.glove_embeddings_index:
            return self.glove_embeddings_index.get(word)
        else:
            if word not in self.dynamic_emb_dict:
                self.dynamic_emb_dict[word] = np.random.uniform(-1, 1, 300)
            return self.dynamic_emb_dict[word]
    
    def get_tag_embedding(self):
        tags_emmbeddings={}
        tags_emmbeddings["O_ANS"]= np.random.uniform(-1, 1, 300)
        tags_emmbeddings["B_ANS"]= np.random.uniform(-1, 1, 300)
        tags_emmbeddings["I_ANS"]= np.random.uniform(-1, 1, 300)
        return tags_emmbeddings
    

    def get_glove_emmbedding_way_1(self,tagged_sentence):
        tags_emmbeddings = {"O_ANS":0,"B_ANS":1,"I_ANS":2}
        context_words_emmbedings = []
        context_tags_emmbedings=[]
        for token_tag in tagged_sentence.split():
            token, tag = token_tag.split('￨')
            context_words_emmbedings.append(self.get_embedding(token))
            context_tags_emmbedings.append(tags_emmbeddings[tag])
        return context_words_emmbedings,context_tags_emmbedings

    def get_glove_emmbedding_way_2(self,tagged_sentence):
        tags_emmbeddings = {"O_ANS":[0,0,0,0,0],"B_ANS":[1,1,1,1,1],"I_ANS":[2,2,2,2,2]}
        context_words_emmbedings = []

        for token_tag in tagged_sentence.split():
            token, tag = token_tag.split('￨')
            context_words_emmbedings.append(np.concatenate((self.get_embedding(token), tags_emmbeddings[tag])))
        return context_words_emmbedings,None
    


    def get_glove_emmbedding_way_3(self, tagged_sentence):
        tags_emmbedings = self.get_tag_embedding()
        context_combined_embeddings = []
        for token_tag in tagged_sentence.split():
            token, tag = token_tag.split('￨')
            word_embedding = self.get_embedding(token)
            tag_embedding = np.array(tags_emmbedings[tag])

            combined_embedding = 0.8 * word_embedding + 0.2 * tag_embedding
            context_combined_embeddings.append(combined_embedding)

        return context_combined_embeddings, None