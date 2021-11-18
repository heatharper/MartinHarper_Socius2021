# HH version September 2021 for python 3 
# -*- coding: utf-8 -*-
# imported via Isaac_Harper_RunAnalysis.py
# make sure all files are in same directory

import csv
import re
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sklearn as sk
import statsmodels.api as sm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import OLSInfluence


class Corpus:

    def __init__(self,
                 data_directory,
                 source_file_path,
                 stopwords_list,
                 citynames,
                 max_features=5000,
                 n_gram=1,
                 sample=None):

        self._source_file_path = source_file_path
        self._data_directory = data_directory
        self.stopwords = stopwords_list
        self.citynames = citynames
        self._n_gram = n_gram
        self.dataframe = pd.read_csv(source_file_path, sep=',', encoding='utf-8')
        self.data_frame = self.dataframe.set_index(['id'])
        self.data_frame.sort_index(inplace=True)
        if sample:
            self.data_frame = self.data_frame.sample(frac=0.8)
        self.size = self.data_frame.count(0)[0]
        self.yes = self.data_frame['yes'].tolist()
        self.IDs = self.data_frame.index.tolist()
        self.fnames = self.data_frame['filename'].tolist()
        self.feature_array = None
        self.v_data_dense = None
        self.vectorizer = TfidfVectorizer(decode_error='ignore', ngram_range= (1, self._n_gram),
            min_df=3, analyzer='word', tokenizer=self.dummy_fun, preprocessor=self.dummy_fun, token_pattern=None)
        self.vectorizer_count = CountVectorizer(decode_error='ignore', ngram_range= (1, self._n_gram),
            min_df=3, analyzer='word', tokenizer=self.dummy_fun, preprocessor=self.dummy_fun, token_pattern=None)
        

# A tokenization function; Make lowercase, tokenize and stem. Filter small words, stop words, non alphabetic.
    def tokenize(self, texts):
        """
        Tokenize function using NLTK tokenizer and Porter stemmer.
        """
        stop_words = list(stopwords.words(self.stopwords))
        with open(self.citynames, encoding='ISO-8859-1') as csv_file:
            reader = csv.reader(csv_file)
            city_name_list = list(reader)
        self.to_remove = [city_name.strip() for city_names in city_name_list for city_name in city_names]
        self.new_stopwords = stop_words 
        # This is a project-specific list of stopwords
        min_length = 3
        filtered_tokens = []
        for text in texts: 
            text = text.replace("\n", " ")
            text = re.sub(r'[^a-zA-Z-]', r" ", text)
            text = re.sub(r" - ", " ", text)
            #text = text.replace("\r", " ")
            text = text.replace("\n", "")
            text = text.lower()
            i = 0
            while i < len(self.to_remove):
                text = text.lower()
                token = self.to_remove[i]
                text = text.replace(token, " ")
                text = text.replace("  ", " ")
                i += 1
            words = map(lambda word: word.lower(), word_tokenize(text))
            words = [word for word in words if word not in self.new_stopwords]
            tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
            p = re.compile('[a-zA-Z]+')
            f_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length, tokens))
            filtered_tokens.append(f_tokens)
        return filtered_tokens
    
    
    def load_data(self):
        """
        Read in text data.
        """
        data = []
        for fn in self.fnames:
            data_dir = self._data_directory if self._data_directory[-1] == '/' else self._data_directory + '/'
            path = data_dir + fn
            with open(path) as f:
                data.append(f.read())
        return data


    # to print top words and top documents for each component 
    def print_top_words_documents(self, model, W, TOPcomponents, feature_names, documents, n_top_words, top_documents):
        for topic_idx, topic in enumerate(model.components_[0:TOPcomponents]):
            print ("Topic %d, positive:" % (topic_idx))
            print (", ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print ([topic[i] for i in topic.argsort()[:-n_top_words -1:-1]])
            print ("Topic %d, negative:" % (topic_idx))
            print (", ".join([feature_names[i]
                        for i in topic.argsort()[:n_top_words - 1:1]]))
            print ([topic[i] for i in topic.argsort()[:n_top_words -1:1]])
            print ("Top docs:")
            top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:top_documents]
            for doc_index in top_doc_indices:
                print (documents[doc_index])
            print ("Bottom docs:")
            bottom_doc_indices = np.argsort( W[:, topic_idx] )[::1][0:top_documents]
            for doc_index in bottom_doc_indices:
                print (documents[doc_index])
    
    def print_top_words(self, model, W, TOPcomponents, feature_names, n_top_words, top_documents):
        for topic_idx, topic in enumerate(model.components_[0:TOPcomponents]):
            print ("Topic %d, positive:" % (topic_idx))
            print (", ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print ("Topic %d, negative:" % (topic_idx))
            print (", ".join([feature_names[i]
                        for i in topic.argsort()[:n_top_words - 1:1]]))

    def dummy_fun(self, doc):
        return doc

    def tfidf_dense(self, text):
        vectorizer = self.vectorizer
        self.v_data = vectorizer.fit_transform(text)
        self.feature_array = np.array(self.vectorizer.get_feature_names())
        self.feature_len = len(self.feature_array)
        self.v_data_dense = self.v_data.toarray()
        tfidf_df = pd.DataFrame(data=self.v_data_dense, columns=self.feature_array, index=self.IDs)
        tfidf_df.index = tfidf_df.index.astype(int)
        tfidf_df.sort_index(inplace=True)
        return tfidf_df

    def count_dense(self, text):
        vectorizer = self.vectorizer_count
        self.v_data = vectorizer.fit_transform(text)
        self.feature_array = np.array(self.vectorizer_count.get_feature_names())
        self.feature_len = len(self.feature_array)
        self.v_data_dense_tf = self.v_data.toarray()
        tf_df = pd.DataFrame(data=self.v_data_dense, columns=self.feature_array, index=self.IDs)
        tf_df.index = tf_df.index.astype(int)
        tf_df.sort_index(inplace=True)
        return tf_df
    

    def LOOCV(self, fitted_model, depvar, indvar):
        influence = OLSInfluence(fitted_model)
        X_hat = OLSInfluence.summary_frame(influence).hat_diag
        predictions = fitted_model.predict(indvar)
        errors = np.asarray(depvar) - np.asarray(predictions)
        mse = ((errors*errors)/((1-X_hat)*(1-X_hat))).sum()*(1/float(len(depvar)))
        return mse
    

    def term_selection(self, feature_len, feature_array, v_data_dense, min_terms):
        output = []
        yes = np.array(self.yes)
        for x in range(0,self.feature_len): # v.data.shape[1] is the n of features
            X = np.array(v_data_dense[:,x]).reshape(-1,1)
            # reshape is necessary to indicate direction 1-feature array
            X = sm.add_constant(X)
            univariate_reg = sm.OLS(yes,X)
            univariate_results = univariate_reg.fit()
            influence = OLSInfluence(univariate_results)
            output = output+[(x, feature_array[x],univariate_results.rsquared ** .5)]
            Y = np.array(output)
        Output = np.array(output, dtype=[('column','int32'),('word', 'S12'),('Beta','float')])
        Output.sort(order='Beta') # Sort in ascending order of abs standardized coeff...
        Output = Output[::-1] # ...and reverse order
        selection = []
        i = min_terms
        print ("\nSelecting Theta by cross-validation...")
        while i < self.size: # v.data.shape[0] is the n of measures
            pca = PCA (n_components=i)
            # select the appropriate columns from v_data_dense
            FirstI = Output[0:i,] # choose first i rows from Output
            # first PCA will be 2 components from top 3 features(sorted by r2), second PCA will be 3 components from top 4 features
            Columns = FirstI['column'] # then for each, look up column
            Temporary = v_data_dense[:,Columns]
            v_data_pca = pca.fit_transform(Temporary) # 1 row per measure, 1 col per component
            # assess fit of regression of yes on first component
            regressor = sm.add_constant(v_data_pca[:,0:1])
            PCA_reg = sm.OLS(yes, regressor)
            results = PCA_reg.fit()
            LOOCV = self.LOOCV(results, yes, regressor)
            loocv = LOOCV
            selection = selection + [(i, loocv)]
            i += 1
        Selection = np.array(selection, dtype=[('terms','int32'),('LOOCV','float')])
        for j in range(1, Selection.shape[0]):
            if Selection[j][1]==min(Selection['LOOCV']):
                Terms = Selection[j][0]
        return Terms, Output, Selection, LOOCV

    
   
    

   
    