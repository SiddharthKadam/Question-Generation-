import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Meteor:
    def __init__(self):
        pass

    
    def evaluate(self,data):
        data["meteor_score"] = 0

        for i,row in data.iterrows():
            data.loc[i,"meteor_score"] =  self.get_meteor_score(row.gen_question,row.question)
        
        return data
    
    
    def get_precision_recall(self,candidate_list,refernce_list):

        total_candidate_list = len(candidate_list)
        total_refernce_list = len(refernce_list)

        sim_count = 0

        for word in candidate_list:
            if word in refernce_list:
                sim_count+=1
                refernce_list.remove(word)
                
        # print("Common Word Found : ",sim_count)
        
        precison = sim_count / total_candidate_list
        recall = sim_count / total_refernce_list

        return precison,recall
    
    def get_chunk_penalty(self,candidate,refernce):
        
        total_candidate = len(candidate)
        
        total_chunk = 0 
        index = 0

        for i,word in enumerate(candidate):
            
            if " ".join(candidate[index:i+1]) not in " ".join(refernce):
                total_chunk+=1
                index = i

        total_chunk+=1
        
        # print("Total Chunk Found : ",total_chunk)

        
        chunk_penalty =     0.5 * (total_chunk/total_candidate)**3
        return chunk_penalty
        
    def get_meteor_score(self,candidate,refernce):
        
        candidate = candidate.split()
        refernce = refernce.split()
        
        chunk_penalty = self.get_chunk_penalty(candidate=candidate,refernce=refernce)
        precison,recall =  self.get_precision_recall(candidate,refernce)
        
        f_mean = (10 * precison * recall) / (recall + 9 * precison)   
        meteor_score =  f_mean * (1-chunk_penalty)

        return meteor_score
    