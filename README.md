# IR_project

## Introduction
This project is part of an Information Retrieval course at BGU University. 

The goal of the project is to implement a search engine for the entire English *Wikipedia* corpus.

The dataset contain over 6 million documents and implemented in *Google Cloud*. 

## Project Structure

The project is divided into 3 main parts:

1. Indexing creation (pre_process) - 
    1. Parsing the corpus -
        we created 5 different parsers for the corpus: Title Parser, Text Parser, anchor Parser, Title stemming Parser, Text stemming Parser. also we calculated Page Rank and Page View for each document.
    2. Creating the index - 
        for each of the parsers above we created an inverted index for the corpus. we used the following data structures:
       1. document frequency - dictionary of the term and the number of documents it appears in.
       2. posting_locs - dictionary mapping each term to the location of the posting file.
       3. term total - dictionary mapping each term to the total number of times it appears in the corpus.
       4. NF (Normalization Factor) - dictionary mapping each document to the length of the document.
       5. we also saved for the body index the length of the tf_idf vector for each document. to ease the calculation of the cosine similarity.
   3. Saving the index - 
       we saved the index in a binary format in GCP bucket.

2. Search Engine (ir_tools) -
    1. Loading the index (ir_tools/inverted_index_gcp) - 
        we loaded the indexes for the search from the project GCP bucket.
    2. Searching for a query (ir_tools/search_frontend) - 
       1. search - the main search function. this function is the main function of the search engine. we have implemented this part using BM25 on the body with stemming, jaccard distance for the title with stemming ranking and Page Rank. each got a weight in the calculation of the relevance score. we also implemented spelling correction word Ngram similarity.
       2. search_body - using cosine similarity on the body index.
       3. search_title - using binary ranking on the title index.
       4. search_anchor - using binary ranking on the anchor index.
       5. search_page_rank - return pre-calculated page rank for each document.
       6. search_page_view - return pre-calculated page view for each document.
      * for the search_body, search_title, search_anchor and search_page_view we read the index from the GCP bucket each query. in order to save memory for the indexes of the main search function.

3. Evaluation - we experienced with the giving train quarries and the evaluation of the search engine. we used the following evaluation metrics:
   1. Recall
   2. MAP@40
   3. reaction time


