# HH version September 2021
# -*- coding: utf-8 -*-
# IM changes July 18, 2019: added year dummies to contextual variables at 173ff.
# imported via Isaac_Harper_RunAnalysis.py
# make sure all files are in same directory

import numpy as np
import pandas as pd
import statsmodels.api as sm
from corpus_PY3 import Corpus
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class PCAModel(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.v_data_H = None
        self.final_num_components = None
        self.comp_df = None
        self.context_df = None
        self.all_df = None
        self.group_var = None

    def ols_model(self, X, modeltype):
        y = self.corpus.yes
        X = sm.add_constant(X)
        self.x = X
        model = sm.OLS(y, X)
        results = model.fit(cov_type = 'cluster', cov_kwds = {'groups': self.group_var})
        print ("\nOLS MODEL: %s" % modeltype)
        print (results.summary())
        fit = self.corpus.LOOCV(results, y, X)
        print ("\nRoot MSE from leave-one-out cross-validation: %s\n" % fit ** .5)
        return results.summary()

    def FE_model(self, X, modeltype):
        y = self.corpus.data_frame['yes']
        X = sm.add_constant(X)
        y_centered = y-y.groupby(self.group_var).transform('mean')+y.mean()
        X_centered = X-X.groupby(self.group_var).transform('mean')+X.mean()
        model = sm.OLS(y_centered, X_centered)
        results = model.fit(cov_type = 'cluster', cov_kwds = {'groups': self.group_var})
        print ("\nFIXED EFFECTS MODEL: %s" % modeltype)
        print (results.summary())
        fit = self.corpus.LOOCV(results, y_centered, X_centered)
        print ("\nRoot MSE from leave-one-out cross-validation: %s\n" % fit ** .5)
        return results.summary()


class Kmeans_PCA(PCAModel):
    def PCA_model_regressor(self, Terms, Output, Selection, LOOCV):
        yes = np.array(self.corpus.yes)
        k = 1
        NewLOOCV = max(Selection['LOOCV'])
        self.pca = PCA (n_components=Terms)
        FirstI = Output[0:Terms,]
        Columns = FirstI['column']
        Temporary = self.corpus.v_data_dense[:,Columns]
        self.tfidf = Temporary
        v_data_pca = self.pca.fit_transform(Temporary)
        self.comps_measures = v_data_pca
        self.v_data_H = self.pca.components_
        self.expl_var = self.pca.explained_variance_
        print ("\nNow selecting number of components by cross validation:")
        while k < Terms:
            regressor = sm.add_constant(v_data_pca[:,0:k])
            PCA_reg = sm.OLS(yes,regressor)
            results = PCA_reg.fit()
            OldLOOCV = NewLOOCV
            NewLOOCV = self.corpus.LOOCV(results, yes, regressor)
            if NewLOOCV <=OldLOOCV:
                print (k, "-component regression yields cross-validation RMSE of %s." % NewLOOCV ** .5)
                k += 1
            else:
                print ("\nOptimal number of components is", k-1)
                regressor = sm.add_constant(v_data_pca[:,0:k-1])
                self.regressor = regressor
                PCA_reg = sm.OLS(yes,regressor)
                results = PCA_reg.fit()
                self.results = results
                print(results.summary())
                self.final_num_components = k - 1
                k = Terms     
                #needed to convert to a list since in python3 the numpy boolean subtract, " - " is depreciated
                params_list = results.params.tolist()          
                # Components have arbitrary sign; re-sign so that regression coefficients are positive
                for i in range(1,len(results.params)):
                    for row in regressor:
                        row[i] = row[i]*((params_list[i]>0) - (params_list[i]<0))
                #Save Component by Term matrix, signed so that regression coefficients are positive
                comp_by_term = self.v_data_H[:regressor.shape[1]-1,:] 
                for i in range(len(results.params)-1):
                    comp_by_term[i] = comp_by_term[i]*((params_list[i+1]>0) - (params_list[i+1]<0))
        return regressor, comp_by_term 
        #regressor is numpy.ndarray with shape (corpus size, constant + N components); constant in col 1
        
    def KMEANS(self, n_clusters=7, min_df=1, max_df=0.95):
        km = KMeans(n_clusters=n_clusters,
                    init='k-means++',
                    max_iter=100,
                    n_init=10,
                    verbose=0,
                    random_state=1)
        km.fit(self.corpus.v_data_dense)
        self.clusters = km.predict(self.corpus.v_data_dense)
        print ('final inertia: %f' % km.inertia_)
        self.ordered_centroids = km.cluster_centers_.argsort()[:, ::-1]
        clusters_df = pd.DataFrame(data=self.clusters, index=self.corpus.IDs, columns=['cluster'])
        #get df of dummy vars from clusters, don't drop first cluster,
        self.cluster_dummy_df = pd.get_dummies(clusters_df.cluster, prefix='cluster', drop_first=False)
        #drop cluster 1 (utility taxes), which will be the reference cateogry
        #to keep in line with original analysis and so that all coefficients are positive
        self.cluster_dummy_df.drop(['cluster_1'], axis=1, inplace=True)
        return self.clusters, self.ordered_centroids
    
    def extract_dataframes(self, regressor, drop_columns, groupby_var, clusters):
        regressor_colsize = int(str(regressor.shape[1]))
        z = ['constant']
        i = 1
        while i < regressor_colsize:
            zz = 'comp %s' %i
            z.append(zz)
            i += 1
        pca_dataframe = pd.DataFrame(data= regressor, index=self.corpus.IDs, columns=z)
        h = 1
        #while h < regressor_colsize:
            #hh = 'comp %s' %h
            #pca_dataframe[hh] = (pca_dataframe[hh]-pca_dataframe[hh].mean())/pca_dataframe[hh].std()
            #h += 1
        #Set up all dataframes WITHOUT constant--will add constant to methods above
        comp_df = pca_dataframe.drop(columns=['constant']) #just components and constant
        context = self.corpus.data_frame
        contextual_df = context.drop(columns=drop_columns) #contextual only
        # Clusters as ordinal var, for graphs; and as dummies for regression
        clusters_df = pd.DataFrame(data=clusters, index=self.corpus.IDs, columns=['clusters']) 
        #drop cluster 1 (utility taxes), which will be the reference cateogry
        clusters_dummy_df = pd.get_dummies(clusters_df.clusters, prefix='cluster', drop_first=False) 
        clusters_dummy_df.drop(['cluster_1'], axis=1, inplace=True)
        clustercon_df = pd.concat([clusters_dummy_df,contextual_df], axis=1)
        #If there is a variable called 'year', convert to year dummies with reference category dropped
        if 'year' in context.columns:
            year_dummies_df = pd.get_dummies(context.year, prefix='yr', drop_first=True)
            context_yrs_df = pd.concat([contextual_df,year_dummies_df], axis=1) 
            comp_yrs_df = pd.concat([comp_df,year_dummies_df],axis=1)
            cluster_yrs_df = pd.concat([clusters_dummy_df,year_dummies_df],axis=1)
            clustercon_yrs_df = pd.concat([clusters_dummy_df,contextual_df,year_dummies_df], axis=1)
            comp_cluster_yrs_df = pd.concat([comp_df,clusters_dummy_df,year_dummies_df],axis=1)
        else: 
            context_yrs_df = contextual_df #otherwise return contextual dataframe without yr dummies
            comp_yrs_df = comp_df
            cluster_yrs_df = clusters_dummy_df
            clustercon_yrs_df = clustercon_df
            comp_cluster_yrs_df = pd.concat([comp_df, clusters_dummy_df], axis=1)
        compcon_df = pd.concat([comp_df, contextual_df], axis=1) #comps + contextual
        compcon_yrs_df = pd.concat([comp_df, context_yrs_df], axis=1) #comps + contextual + yr dummies
        comp_clusters_df = pd.concat([comp_df, clusters_dummy_df], axis=1)
        comp_1cluster_df = pd.concat([comp_df, clusters_df], axis=1) 
        pca_cluster1_df = comp_1cluster_df.astype({"clusters": float}) #comps + ordinal cluster for scatter plot
        all_df = pd.concat([compcon_df, clusters_dummy_df], axis=1)
        all_yrs_df = pd.concat([compcon_yrs_df, clusters_dummy_df], axis=1)
        self.all_dataframe = all_df
        self.group_var = self.corpus.data_frame[groupby_var]
        return comp_df, comp_yrs_df, clusters_dummy_df, cluster_yrs_df, contextual_df, context_yrs_df, clustercon_df, clustercon_yrs_df, comp_1cluster_df, comp_clusters_df, comp_cluster_yrs_df, compcon_df, compcon_yrs_df, all_df, all_yrs_df

 


      

        
        
    
        
    
    
        
    
        

