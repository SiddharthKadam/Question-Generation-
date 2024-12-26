import numpy as np
from named_entity_recognition  import NER
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


class ModelDataPreparation:
    def __init__(self,QUES_ANS_PATH,CONTEXT_PATH):
        self.QUES_ANS_PATH = QUES_ANS_PATH
        self.CONTEXT_PATH = CONTEXT_PATH
        self.load_data()
        self.manipulate_data()
        self.prepare_model_data()


    def load_data(self):
        with open(self.QUES_ANS_PATH, 'rb') as f:
            self.q_a = pickle.load(f)
        with open(self.CONTEXT_PATH, 'rb') as f:
            self.context = pickle.load(f)

    def manipulate_data(self):
        self.context= self.context[["context_id","context"]]
        self.q_a = self.q_a[self.q_a["is_impossible"]==False]
        self.q_a["context"]=None
        
        
        for i, row in self.q_a.iterrows():
            self.q_a.loc[i, "context"] = self.context.loc[self.context["context_id"] == row.context_id, "context"].values[0]
        def add_question_tag(question):
            return 'starttag ' + question + ' endtag'
        self.q_a["question"]=self.q_a["question"].apply(add_question_tag)

        def add_punctuation_space(row):
            row=row.replace("."," . ")
            row = row.replace(","," , ") 
            row = row.replace("?"," ? ")
            
            # row = row.replace('['," [ ")
            # row = row.replace(']'," ] ")
            
            # row = row.replace('('," ( ")
            # row = row.replace('('," ) ")

            # row = row.replace('{'," { ")
            # row = row.replace('}'," } ")

            row = row.replace(';',"") 
            row = row.replace('"',"") 
            row = row.replace(':',"")
             
            return row
        
        def lower_sentence(text):
            return text.lower()
        
        def lower_answers(ans_list):
            return [ans.lower() for ans in ans_list]
        
        self.q_a["question"]=self.q_a["question"].apply(add_punctuation_space)
        self.q_a["context"]=self.q_a["context"].apply(add_punctuation_space)
        self.q_a["context"]=self.q_a["context"].apply(lower_sentence)
        self.q_a["question"]=self.q_a["question"].apply(lower_sentence)

        self.q_a["ans_list"] = self.q_a["ans_list"].apply(lower_answers)
    def add_question_tag(question):
        return 'starttag ' + question + ' endtag'

    
    def prepare_model_data(self):
        self.q_a["context_list"]=None
        self.q_a["combined_context"] =  None
        from bio_notation import BIO
        bio = BIO()
        
        for i,row in self.q_a.iterrows():
            temp_context_list = []
            text = ""
            answers=list(set(row["ans_list"]))
            # for ans in answers:
            #     bio_text = bio.convert_BIO(row.context,ans)
            #     temp_context_list.append(bio_text)
            #     text+=" "+ bio_text +" "
            bio_text = bio.convert_BIO(row.context,answers[0])
            temp_context_list.append(bio_text)
            text+=" "+ bio_text +" "


            self.q_a.loc[i,"combined_context"] = row.question+" "+text
            
            self.q_a.at[i,"context_list"] =  temp_context_list


class EmbbedingMatrix:
    def __init__(self,q_a,GLOVE_FILE_PATH,vocab_size=10000):
        self.q_a = q_a
        self.vocab_size =vocab_size
        from glove_embeddings import GLOVE_EMBEDDINGS
        # update this path to yours 
        self.glove =GLOVE_EMBEDDINGS(GLOVE_FILE_PATH)
        self.make_emmbedding_matric()


    def make_emmbedding_matric(self):
        tokenizer_text = Tokenizer(num_words=self.vocab_size, oov_token='<unk>')
        tokenizer_text.fit_on_texts(self.q_a["combined_context"])
        # self.context_seq = tokenizer_text.texts_to_sequences(self.q_a["context"].values)
        # self.question_seq = tokenizer_text.texts_to_sequences(self.q_a["question"].values)
        
        embedding_dim = 300
        embedding_matrix = np.zeros((self.vocab_size, embedding_dim))

        # Handling Special Token like unk , start, end
        embedding_matrix[tokenizer_text.word_index["<unk>"]] = np.random.normal(size=(embedding_dim,))  # Random initialization
        embedding_matrix[tokenizer_text.word_index["starttag"]] = np.random.normal(size=(embedding_dim,))  # Random initialization
        embedding_matrix[tokenizer_text.word_index["endtag"]] = np.random.normal(size=(embedding_dim,))  # Random initialization


        for word, i in tokenizer_text.word_index.items():
            if i < self.vocab_size:
                embedding_vector = self.glove.glove_embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        
        self.tokenizer_text = tokenizer_text

        self.embedding_matrix = embedding_matrix

        self.create_data_sequences()
    def create_data_sequences(self):
        questions_list = []
        context_list = []
        for i,row in self.q_a.iterrows():
            for j in row.context_list:
                questions_list.append(row.question)
                context_list.append(j)
            
        self.df = pd.DataFrame({
            'question': questions_list,
            'context': context_list
        })


        self.context_seq = self.tokenizer_text.texts_to_sequences(self.df["context"].values)
        self.question_seq = self.tokenizer_text.texts_to_sequences(self.df["question"].values)
        
                # Pad sequences to equal length
        self.max_len_context = max(len(seq) for seq in self.context_seq)
        self.max_len_question = max(len(seq) for seq in self.question_seq)

        self.context_seq = pad_sequences(self.context_seq, maxlen=self.max_len_context, padding='post')
        self.question_seq = pad_sequences(self.question_seq, maxlen=self.max_len_question, padding='post') 


# import numpy as np
# import pandas as pd
# from keras.preprocessing.sequence import pad_sequences

# # SimpleTokenizer class for tokenization using dictionaries
# class SimpleTokenizer:
#     def __init__(self, vocab_size, oov_token='<unk>'):
#         self.vocab_size = vocab_size
#         self.oov_token = oov_token
#         self.word_index = {oov_token: 1}  # OOV token will have index 1
#         self.index_word = {1: oov_token}
#         self.glove = None

#     def fit_on_texts(self, texts):
#         word_counts = {}
#         for text in texts:
#             for word in text.split():
#                 word_counts[word] = word_counts.get(word, 0) + 1
        
#         sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size-1]
        
#         # Build word to index mapping
#         for i, (word, _) in enumerate(sorted_words, start=2):  # Indexing starts at 2
#             self.word_index[word] = i
#             self.index_word[i] = word

#     def texts_to_sequences(self, texts):
#         sequences = []
#         for text in texts:
#             sequences.append([self.word_index.get(word, self.word_index[self.oov_token]) for word in text.split()])
#         return sequences

#     def set_glove_embeddings(self, glove_embeddings):
#         self.glove = glove_embeddings

#     def get_glove_embeddings(self):
#         return self.glove


# # EmbeddingMatrix class to create embedding matrix using GloVe embeddings
# class EmbbedingMatrix:
#     def __init__(self, q_a, GLOVE_FILE_PATH, vocab_size=10000, embedding_dim=300):
#         self.q_a = q_a
#         self.vocab_size = vocab_size
#         self.embedding_dim = embedding_dim
#         self.glove = self.load_glove_embeddings(GLOVE_FILE_PATH)
#         self.tokenizer_text = SimpleTokenizer(vocab_size=self.vocab_size)
#         self.tokenizer_text.fit_on_texts(self.q_a["combined_context"])  # Assuming "combined_context" column exists
#         self.tokenizer_text.set_glove_embeddings(self.glove)
#         self.make_embedding_matrix()

#     def load_glove_embeddings(self, file_path):
#         glove_embeddings = {}
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 values = line.split()
#                 word = values[0]
#                 vector = np.asarray(values[1:], dtype='float32')
#                 glove_embeddings[word] = vector
#         return glove_embeddings

#     def make_embedding_matrix(self):
#         # Initialize the embedding matrix with zeros
#         embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))

#         # Handle special tokens like <unk>, starttag, endtag
#         embedding_matrix[self.tokenizer_text.word_index.get("<unk>", 1)] = np.random.normal(size=(self.embedding_dim,))
#         embedding_matrix[self.tokenizer_text.word_index.get("starttag", 2)] = np.random.normal(size=(self.embedding_dim,))
#         embedding_matrix[self.tokenizer_text.word_index.get("endtag", 3)] = np.random.normal(size=(self.embedding_dim,))

#         # Populate the matrix with GloVe vectors for known words
#         for word, i in self.tokenizer_text.word_index.items():
#             if i < self.vocab_size:
#                 embedding_vector = self.glove.get(word)
#                 if embedding_vector is not None:
#                     embedding_matrix[i] = embedding_vector

#         self.embedding_matrix = embedding_matrix
#         self.create_data_sequences()

#     def create_data_sequences(self):
#         questions_list = []
#         context_list = []

#         # Create pairs of questions and contexts
#         for i, row in self.q_a.iterrows():
#             for j in row.context_list:
#                 questions_list.append(row.question)
#                 context_list.append(j)

#         self.df = pd.DataFrame({
#             'question': questions_list,
#             'context': context_list
#         })

#         # Convert the questions and contexts into sequences of indices
#         self.context_seq = self.tokenizer_text.texts_to_sequences(self.df["context"].values)
#         self.question_seq = self.tokenizer_text.texts_to_sequences(self.df["question"].values)

#         # Pad sequences to equal length
#         self.max_len_context = max(len(seq) for seq in self.context_seq)
#         self.max_len_question = max(len(seq) for seq in self.question_seq)

#         self.context_seq = pad_sequences(self.context_seq, maxlen=self.max_len_context, padding='post')
#         self.question_seq = pad_sequences(self.question_seq, maxlen=self.max_len_question, padding='post')

#     def get_embedding_matrix(self):
#         return self.embedding_matrix

#     def get_data_sequences(self):
#         return self.context_seq, self.question_seq



