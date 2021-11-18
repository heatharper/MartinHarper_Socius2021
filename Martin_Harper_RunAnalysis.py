# HH version September 2021
# !/usr/bin/env python
# coding: utf-8

from corpus_PY3 import Corpus
from SupPCA_cluster_PY3 import Kmeans_PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


# indicate path for contextual information, directory for text files, and stopword list
# UPDATE PATH INFORMATION 
path = "/Users/heatharper/desktop/MartinHarper_Socius2021_RunFiles/"
fnames_covariates = path + "City848_Impartial_All_forPython_revised.csv"
directory = "CITY_DATA"
stopwords_list = path + "english+"
citynames = path + "city names.csv"
corpus = Corpus(source_file_path=fnames_covariates, data_directory=directory, stopwords_list=stopwords_list, citynames=citynames)
print('corpus size:', corpus.size)

# load data, tokenize data, extract vectorized dataframe (with ballot IDs as sorted index, tfidf weights, and features as column names), 
# AND extract dense matrix of tfidf weights
data = corpus.load_data()

# preprocess by replacing the abbreviation 'UUT' with 'utility users tax'; extract dense matrix
for i in range(0, len(data)):
    data[i] = data[i].replace("UUT","Utility User\'s Tax")
    data[i] = data[i].replace("TOT","Transient Occupancy Tax")
token_data = corpus.tokenize(data)
tfidf_matrix = corpus.tfidf_dense(token_data)
v_data_dense = corpus.v_data_dense

# print vocabulary length and selected features
feature_array = corpus.feature_array
print('feature length:', corpus.feature_len)
terms, output, selection, LOOCV = corpus.term_selection(corpus.feature_len, feature_array, v_data_dense, 20)
print ("\n %s-term solution minimizes cross-validation error." % terms)

# initiate class & regressor method
analyze = Kmeans_PCA(corpus)
regressor, comp_by_term = analyze.PCA_model_regressor(terms, output, selection, LOOCV)

# run kmeans clustering 
n_clusters=5
clusters, ordered_centroids = analyze.KMEANS(n_clusters=n_clusters)

# print top words for derived clusters
for i in range(n_clusters):
    outstr = "\n CLUSTER %d: " % i
    outstr += ' '.join([feature_array[idx] for idx in ordered_centroids[i, :20]])
    # here, loop through the top 10 features and join them together
    print (outstr)

# write csv, for each cluster: top terms, docs associated
f = open('kmean_output.csv', 'w')
f.write(str(n_clusters)+'\n')
for i in range(n_clusters):
    outstr = "Cluster %d: " % i
    outstr += ' '.join([feature_array[idx] for idx in ordered_centroids[i, :20]])
    f.write(outstr+'\n')
    f.write("docs' names" + '\n')
    for n in np.squeeze(np.asarray(np.where(clusters == i))):
        f.write(corpus.fnames[n] + '\n')
f.close()

# columns to drop (those don't want included in analysis); variable name to use for fixed effects; 
# implement method to get all dataframes for scatter plot and regressions
df_drop_list = ['yes', 'filename', 'IDcity', 'year']
groupby_var = 'IDcity'
comp_df, comp_yrs_df, clusters_dummy_df, cluster_yrs_df, contextual_df, context_yrs_df, clustercon_df, clustercon_yrs_df, \
    comp_1cluster_df, comp_clusters_df, comp_cluster_yrs_df, compcon_df, compcon_yrs_df, all_df, all_yrs_df \
    = analyze.extract_dataframes(regressor, df_drop_list, groupby_var, clusters)

# rename cluster columns for scatter
scatter = comp_1cluster_df.replace({'clusters': {0:'Business', 1:'Utility', 2:'Sales', 
                                                 3:'Parcel', 4:'Hotel'}})
# get multidimensional scaling scatter 
ax = sns.lmplot('comp 1', 'comp 2', data=scatter, hue='clusters', aspect=1.5, fit_reg=False)
ax.set(xlabel='risk pooling', ylabel='community orientation')
plt.show()

scatter_og = comp_1cluster_df.rename(columns={'comp 1':'risk pooling',
                          'comp 2':'community orientation'})
i = 'all other data'
scatter0 = scatter_og.replace({'clusters': {0:'Business', 1:i, 2:i , 3:i, 4:i}}).sort_values(by=['clusters'], ascending=False)
scatter1 = scatter_og.replace({'clusters': {0:i, 1: 'Utility', 2:i , 3:i, 4:i}}).sort_values(by=['clusters'], ascending=False)
scatter2 = scatter_og.replace({'clusters': {0:i, 1:i, 2: 'Sales', 3:i, 4:i}}).sort_values(by=['clusters'], ascending=False)
scatter3 = scatter_og.replace({'clusters': {0:i, 1:i, 2:i , 3:'Parcel', 4: i}}).sort_values(by=['clusters'], ascending=False)
scatter4 = scatter_og.replace({'clusters': {0:i, 1:i, 2:i , 3:i, 4: 'Hotel'}}).sort_values(by=['clusters'], ascending=False)

fig, axes = plt.subplots(2, 3, figsize=(10, 7), sharex='col', sharey='row')
sns.despine(left=True)
sns.scatterplot('risk pooling', 'community orientation', data=scatter0, hue='clusters', hue_order=['Business', i], palette=['royalblue', 'grey'], legend=False, ax=axes[0, 0])
sns.scatterplot('risk pooling', 'community orientation', data=scatter1, hue='clusters', hue_order=['Utility', i], palette=['royalblue', 'grey'], legend=False, ax=axes[0, 1])
sns.scatterplot('risk pooling', 'community orientation', data=scatter2, hue='clusters', hue_order=['Sales', i], palette=['royalblue', 'grey'], legend=False, ax=axes[0, 2])
sns.scatterplot('risk pooling', 'community orientation', data=scatter3, hue='clusters', hue_order=['Parcel', i], palette=['royalblue', 'grey'], legend=False, ax=axes[1, 0])
sns.scatterplot('risk pooling', 'community orientation', data=scatter4, hue='clusters', hue_order=['Hotel', i], palette=['royalblue', 'grey'], legend=False, ax=axes[1, 1])
sns.scatterplot('risk pooling', 'community orientation',  data=scatter_og, color='grey', legend=False, ax=axes[1, 2])

a0 = fig.axes[0]
a0.set_title('Cluster 1: Business License Tax')
a1 = fig.axes[1]
a1.set_title('Cluster 2: Utility Users Tax')
a2 = fig.axes[2]
a2.set_title('Cluster 3: Sales Tax')
a3 = fig.axes[3]
a3.set_title('Cluster 4: Parcel Tax')
a4 = fig.axes[4]
a4.set_title('Cluster 5: Hotel Tax')
a5 = fig.axes[5]
a5.set_title('All taxes')
plt.xlabel('risk Pooling')
plt.tight_layout()
plt.show()

# run all OLS models and print output
ols_comps = analyze.ols_model(X=comp_df, modeltype= 'Components Only')
ols_cluster = analyze.ols_model(X=clusters_dummy_df, modeltype= 'Clusters Only')
ols_context = analyze.ols_model(X=contextual_df, modeltype= 'Context only')
ols_cluster_context = analyze.ols_model(X=clustercon_df, modeltype= 'Clusters and Context')
ols_comps_cluster = analyze.ols_model(X=comp_clusters_df, modeltype= 'Components and Clusters')
ols_comps_context = analyze.ols_model(X=compcon_df, modeltype = 'Components and Context')
ols_fullmodel = analyze.ols_model(X=all_df, modeltype= 'Full Model')

# run all fixed effects models and print output
FE_comps = analyze.FE_model(X=comp_yrs_df, modeltype= 'Components Only')
FE_cluster = analyze.FE_model(X=cluster_yrs_df, modeltype= 'Clusters Only')
FE_context = analyze.FE_model(X=context_yrs_df, modeltype= 'Context only')
FE_cluster_context = analyze.FE_model(X=clustercon_yrs_df, modeltype= 'Clusters and Context')
FE_comps_cluster = analyze.FE_model(X=comp_cluster_yrs_df, modeltype= 'Components and Clusters')
FE_comps_context = analyze.FE_model(X=compcon_yrs_df, modeltype = 'Components and Context')
FE_fullmodel = analyze.FE_model(X=all_yrs_df, modeltype= 'Full Model')

# create and print figures with highlighted terms
Components = pd.DataFrame(np.transpose(comp_by_term),columns=['comp1','comp2'])
terms_og = output[:comp_by_term.shape[1]].tolist()
terms = np.char.decode(terms_og, encoding ='utf-8')
TokenLabels = pd.DataFrame([row[1] for row in terms],columns=['term'])
Figure1_data = pd.concat([TokenLabels,Components], axis=1)

# set highlighted terms 
highlighted_terms = ['cardroom','citi','electr','gener','government','insur','marijuana','medic','non-resident','paramed','parcel','percent','residenti','special','telephon','town','user','util','water']
sns.set(rc={'figure.figsize':(10,6)}, style='ticks')
Figure_1 = sns.scatterplot('comp1', 'comp2', data=Figure1_data, facecolors='none')
for item in range(0,Figure1_data.shape[0]):
    Figure_1.text(Figure1_data.comp1[item],Figure1_data.comp2[item],Figure1_data.term[item],horizontalalignment='center', 
     size='small', color='gray', weight='light')
for bolditem in range(0,Figure1_data.shape[0]):
    if Figure1_data.term[bolditem] in highlighted_terms:
        Figure_1.text(Figure1_data.comp1[bolditem],Figure1_data.comp2[bolditem],Figure1_data.term[bolditem],horizontalalignment='center', 
        size='small', color='black', weight='medium')
Figure_1.set(xlabel='risk pooling', ylabel='community orientation')
plt.show()

# create new columns--first multiply each by 10 and by 100; then compute inverse hyperbolic sine function 
# for original values, *10, and *100
Components['comp1_10'] = Components['comp1'].multiply(other = 10)
Components['comp2_10'] = Components['comp2'].multiply(other = 10)
Components['comp1_100'] = Components['comp1'].multiply(other = 100)
Components['comp2_100'] = Components['comp2'].multiply(other = 100)
Components['comp1_ihs'] = Components['comp1'].apply(lambda x: math.asinh(x))
Components['comp2_ihs'] = Components['comp2'].apply(lambda x: math.asinh(x))
Components['comp1_ihs_10'] = Components['comp1_10'].apply(lambda x: math.asinh(x))
Components['comp2_ihs_10'] = Components['comp2_10'].apply(lambda x: math.asinh(x))
Components['comp1_ihs_100'] = Components['comp1_100'].apply(lambda x: math.asinh(x))
Components['comp2_ihs_100'] = Components['comp2_100'].apply(lambda x: math.asinh(x))
print (Components)

# create scatter plot for values computed using inverse hyperbolic sine function, to deal with outlier values like util and user
Components2 = pd.DataFrame({'comp1':Components.comp1_ihs, 'comp2':Components.comp2_ihs})
Figure2_data=pd.concat([TokenLabels,Components2], axis=1)

highlighted_terms=['cardroom','citi','electr','gener','government','insur','marijuana','medic','non-resident','paramed','parcel','percent','residenti','special','telephon','town','user','util','water']
sns.set(rc={'figure.figsize':(10,6)}, style='ticks')
Figure_2 = sns.scatterplot('comp1', 'comp2', data=Figure2_data, facecolors='none')
for item in range(0,Figure2_data.shape[0]):
    Figure_2.text(Figure2_data.comp1[item],Figure2_data.comp2[item],Figure2_data.term[item],horizontalalignment='center', 
     size='small', color='gray', weight='light')
for bolditem in range(0,Figure2_data.shape[0]):
    if Figure2_data.term[bolditem] in highlighted_terms:
        Figure_2.text(Figure2_data.comp1[bolditem],Figure2_data.comp2[bolditem],Figure2_data.term[bolditem],horizontalalignment='center', 
        size='small', color='black', weight='medium')
Figure_2.set(xlabel='risk pooling', ylabel='community orientation')
plt.show()

# create scatter plot for *10 inverse hyperbolic sine function values
Components10 = pd.DataFrame({'comp1':Components.comp1_ihs_10, 'comp2':Components.comp2_ihs_10})
Figure3_data=pd.concat([TokenLabels,Components10], axis=1)

highlighted_terms=['cardroom','citi','electr','gener','government','insur','marijuana','medic','non-resident','paramed','parcel','percent','residenti','special','telephon','town','user','util','water']
sns.set(rc={'figure.figsize':(10,6)}, style='ticks')
Figure_3 = sns.scatterplot('comp1', 'comp2', data=Figure3_data, facecolors='none')
for item in range(0,Figure3_data.shape[0]):
    Figure_3.text(Figure3_data.comp1[item],Figure3_data.comp2[item],Figure3_data.term[item],horizontalalignment='center', 
     size='small', color='gray', weight='light')
for bolditem in range(0,Figure3_data.shape[0]):
    if Figure3_data.term[bolditem] in highlighted_terms:
        Figure_3.text(Figure3_data.comp1[bolditem],Figure3_data.comp2[bolditem],Figure3_data.term[bolditem],horizontalalignment='center', 
        size='small', color='black', weight='medium')
Figure_3.set(xlabel='risk pooling', ylabel='community orientation')
plt.show()

# create scatter plot for *100 inverse hyperbolic sine function values
Components100 = pd.DataFrame({'comp1':Components.comp1_ihs_100, 'comp2':Components.comp2_ihs_100})
Figure4_data=pd.concat([TokenLabels,Components100], axis=1)

highlighted_terms=['cardroom','citi','electr','gener','government','insur','marijuana','medic','non-resident','paramed','parcel','percent','residenti','special','telephon','town','user','util','water']
sns.set(rc={'figure.figsize':(10,6)}, style='ticks')
Figure_4 = sns.scatterplot('comp1', 'comp2', data=Figure4_data, facecolors='none')
for item in range(0,Figure4_data.shape[0]):
    Figure_4.text(Figure4_data.comp1[item],Figure4_data.comp2[item],Figure4_data.term[item],horizontalalignment='center', 
     size='small', color='gray', weight='light')
for bolditem in range(0,Figure4_data.shape[0]):
    if Figure4_data.term[bolditem] in highlighted_terms:
        Figure_4.text(Figure4_data.comp1[bolditem],Figure4_data.comp2[bolditem],Figure4_data.term[bolditem],horizontalalignment='center', 
        size='small', color='black', weight='medium')
Figure_4.set(xlabel='risk pooling', ylabel='community orientation')
plt.show()

# extract components x documents matrix
comps_docs = pd.DataFrame(data=analyze.comps_measures, index=corpus.IDs)
comps_docs.to_csv("City848_comp_by_docs.csv")

# extract terms x components matrix 
terms_comps = pd.DataFrame(data=analyze.v_data_H.T, index=TokenLabels)
terms_comps.to_csv("City848_terms_by_comps.csv")


