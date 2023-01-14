import math
from inverted_index_gcp import *
import heapq

class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    path_to_posting_list: path to posting list

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, path_to_posting_list: str, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        if "postings_title_stemming/" == path_to_posting_list:
            self.N = 6348910
            self.AVGDL = 16035483 / self.N
        if "postings_text_stemming/" == path_to_posting_list:
            self.N = 6348910
            self.AVGDL = 2028630613 / self.N
        else:
            self.N = len(index.NF)
            self.AVGDL = sum(index.NF.values()) / self.N
        self.path_to_posting_list = path_to_posting_list

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

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            n_ti = self.index.df.get(term, 0)
            idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
        return idf

    def search(self, query, N=None):
        """
        This function calculate the bm25 score for given query and document.
        This function return a sorted list of rankings from the best to the worst only of relevent candidates as the following tuples:
        (doc_id, score)

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        list of doc_id's sorted by bm25 score: [doc_id1, ....]
        """
        mini_df = {token: dict(self.read_posting_list(token)) for token in query}

        # build relevent documents set (contain docs with words in commens with query)
        candidate_docs = set()
        candidate_docs_temp = [id for id in mini_df.values()]
        for i in candidate_docs_temp:
            for x in i.keys():
                candidate_docs.add(x)

        # calculate result for each doc
        res = {}
        idf_q = self.calc_idf(query)
        for doc_id in candidate_docs:
            score = 0.0
            for qi in query:
                score += idf_q[qi] * mini_df[qi].get(doc_id, 0) * (self.k1 + 1) / (
                        mini_df[qi].get(doc_id, 0) + self.k1 * (
                        1 - self.b + self.b * self.index.NF[doc_id] / self.AVGDL))
            res[doc_id] = score

        # sort and return the results
        return sorted(res.items(), key=lambda x: x[1], reverse=True)[:N]
