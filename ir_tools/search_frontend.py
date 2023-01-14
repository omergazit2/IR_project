import nltk
from flask import Flask, request, jsonify
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
from BM25 import *
from tf_idf_cosine import *
from inverted_index_gcp import *
import re
import pickle
from nltk.corpus import words
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
correct_words = words.words()


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

def bucket_pkl_read(blob_name, bucket_name = "indexes_wikipidia_ir_project"):
    """read a pkl file from a bucket"""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your new GCS object
    # blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    with blob.open("rb") as f:
        file = pickle.load(f)
    return file
        
        
app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

stemmer = PorterStemmer()
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


# read title dict {(dockID, title) ... }
title_dict = bucket_pkl_read(blob_name = "id_to_title.pickle")

# read pagerank dict {(dockID, pagerank) ... }
page_rank = bucket_pkl_read(blob_name = "PageRank.pickle")

# read the tf-idf vector len of the documents {(docID, doc tf-idf vector len)}
tfidf_vec_len = bucket_pkl_read(blob_name = "tfidf_vec_len.pickle")

# read body index with stemming
body_index_stemming = InvertedIndex()
body_index_stemming = body_index_stemming.read_index("text_index_stemming.pkl")

def jaccard(s1, s2):
    """
    param:
    -------------
    s1,s2 list of tokenized words
    return: jaccard score fore similarity
    """
    s1 = set(s1)
    s2 = set(s2)
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    return len(intersection)/ len(union)

def tokenizer_stemming_no_stopwords_removal(text):
    """
    :param text: string
    :return: tokenized quarry with out stop words removal
    """
    return [stemmer.stem(token.group()) for token in RE_WORD.finditer(text.lower())]

def tokenizer_stemming(text):
    """
    :param text: string
    :return: tokenized quarry after porter stemmer
    """
    return [stemmer.stem(token.group()) for token in RE_WORD.finditer(text.lower()) if stemmer.stem(token.group()) not in all_stopwords]

def tokenizer(text):
    """
    :param text: string
    :return: tokenized quarry from assignment 3
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [token for token in tokens if token not in all_stopwords]


def read_posting_list(w, index, path_to_posting_list):
    """
    function return posting list for spesific token w.
    -----------
      w: token from corpus
      index: inverted index
      path_to_posting_list: the bucket path to the diractory holding the posting lists
    -----------
    returns list [(doc_id, tf) ... ]
    example: [(12, 2), (358, 1), ...]
    """
    if index.df.get(w, None) is None:  # if the word is not on corpus return {}
        return {}
    with closing(MultiFileReader(path_to_posting_list)) as reader:
        locs = index.posting_locs[w]
        b = reader.read(locs, index.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(index.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenizer_stemming(query)
    unknown_words = [token for token in tokens if
                     token not in body_index_stemming.df and token not in all_stopwords]  # words that are not in our corpus

    for incorrect_word in unknown_words:  # use ngram to find the closest correct word
        temp = [(jaccard_distance(set(ngrams(incorrect_word, 2)), set(ngrams(w, 2))), w) for w in correct_words if
                w[0] == incorrect_word[0]]
        tokens.append(stemmer.stem(
            sorted(temp, key=lambda val: val[0])[0][1]))  # add the most similar correct word to the tokens list

    # replace unknown word with the most close word known
    for incorrect_word in unknown_words:
        tokens.remove(incorrect_word)

    bm = BM25_from_index(body_index_stemming, "postings_text_stemming/")
    f1 = lambda x: x if x < 1 else 10  # if title is almost identical by jacard similarity - give 10 times more score

    # calculate score combination between page rank, title and body. then sort
    res = sorted([(x[0], 0.35 + jaccard(tokenizer_stemming_no_stopwords_removal(title_dict[x[0]]),
                                        tokenizer_stemming_no_stopwords_removal(query)),
                   math.log10(page_rank[x[0]] + 1), x[1]) for x in bm.search(tokens, 300)],
                 key=lambda t: f1(t[1]) * t[2] * t[3], reverse=True)  # t1 jacard title t2 pagerank t3 body bm25 score
    res = [(x[0], title_dict[x[0]]) for x in res[:20]]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body(): 
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # read body index
    body_index = InvertedIndex()
    body_index = body_index.read_index("text_index.pkl")
    tokens = tokenizer(query)
    tf = TfIdfCosine(body_index, "postings_text/", tfidf_vec_len)
    res = tf.search(tokens, 100)
    res = [(x, title_dict[x]) for x in res]
    del body_index
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # read title index
    title_index = InvertedIndex()
    title_index = title_index.read_index("title_index.pkl")
    tokens = tokenizer(query)
    result = Counter()
    for token in tokens:
        result.update(map(lambda x: x[0], read_posting_list(token, title_index, "postings_title/")))  # for each word in tokens count 1 time every docID
    res = list(map(lambda x: (x[0], title_dict[x[0]]), result.most_common()))  # map to sorted list of [(docID, docTitle)....]
    del title_index
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # read anchor_text index
    ancor_index = InvertedIndex()
    ancor_index = ancor_index.read_index("ancor_index.pkl")
    tokens = tokenizer(query)
    result = Counter()
    for token in tokens:
        result.update(map(lambda x: x[0],
                          read_posting_list(token, ancor_index, "postings_ancor/")))  # for each word in tokens count 1 time every docID
    res = list(
        map(lambda x: (x[0], title_dict[x[0]]), result.most_common()))  # map to sorted list of [(docID, docTitle)....]
    del ancor_index
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [page_rank.get(int(wiki_id), 0) for wiki_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    # read pageviwe dict {(dockID, pageviwe) ... }
    page_view = bucket_pkl_read(blob_name="pageviews-202108-user.pkl")
    res = [page_view.get(int(wiki_id), 0) for wiki_id in wiki_ids]
    del page_view
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
