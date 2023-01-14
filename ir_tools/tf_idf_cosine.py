import math
from collections import Counter
import numpy as np
import pandas as pd
from inverted_index_gcp import *
import heapq

class TfIdfCosine:
    def __init__(self, index, path_to_posting_list, tfidf_vec_len):
        """
        param:
        -------
        index: inverted index
        path_to_posting_list: string, path to where the index posting list is saved on the bucket
        tfidf_vec_len: dict of {docID: vec_len, ...}
         
        """
        self.index = index
        self.path_to_posting_list = path_to_posting_list
        self.tfidf_vec_len = tfidf_vec_len

    def get_top_n(self, sim_dict, N=None):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores

        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

        N: Integer (how many documents to retrieve). By default N = 3

        Returns:
        -----------
        a ranked list of doc_id sorted by the score in the length of N.
        """
        if N is not None:
            if len(sim_dict) >= N:
                
                sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]
                return heapq.nlargest(N, sim_dict, key=sim_dict.get)
            else:
                return [x[0] for x in sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)]
        return [x[0] for x in sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)]

    def search(self, quarry, N=None):
        """
        :param quarry:  list of tokenized quarry. example: ["blue", "sky", "today"]
        :param quarry:  Number of results to return of None - if None return all
        :return: a ranked list of sorted doc_id's in the length of N.
        """
        quarry = [w for w in quarry if w in self.index.df]
        
        if len(quarry) == 0:
            return []
        
        # get tf_idf values for quarry 
        q_len = len(quarry)
        quarry = Counter(quarry)
        tf_idf_Q = {}
        for w, tf in quarry.items():
            tf_idf_Q[w] = (tf/q_len) * math.log2(len(self.index.NF)/ self.index.df[w])
        Q_vec_len = math.sqrt(sum([x**2 for x in tf_idf_Q.values()]))
        
        #calculate cosine similarity
        res_dic = {}
        for word_Q, w_ij_Q in tf_idf_Q.items(): # for each word in quarry gather all the relevent candidate and calculate its w_ij and then cosine similarity
            for docID_D, tf in self.read_posting_list(word_Q):
                w_ij_D = (tf/ self.index.NF[docID_D]) * math.log2(len(self.index.NF)/ self.index.df.get(word_Q, 1))
                res_dic[docID_D] = res_dic.get(docID_D, 0) + w_ij_D * w_ij_Q
        
        # normalize the result by the vectors len 
        res_dic_normalized = {}
        for docID, sim in res_dic.items():
            res_dic_normalized[docID] = sim/ (self.tfidf_vec_len[docID] * Q_vec_len)
            
        return self.get_top_n(res_dic_normalized, N)
    
    def read_posting_list(self, w):
        """
        function return posting list for spesific token w.
        -----------
          w: token from corpus
        -----------
        returns list [(doc_id, tf) ... ]
        example: [(12, 2), (358, 1), ...]
        """
        if self.index.df.get(w, None) is None:  # if the word is not on corpus return {}
            return {}
        with closing(MultiFileReader(self.path_to_posting_list)) as reader:
            locs = self.index.posting_locs[w]
            b = reader.read(locs, self.index.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(self.index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list
    