This directory contains all the files necessary to reproduce the analysis presented in "What makes a tax policy popular? Predicting referendum votes from policy text" published in Socius 2021. 



For questions, contact Isaac Martin (iwmartin@ucsd.edu) or Heather Harper (hmharper@ucsd.edu).



This RunFiles directory contains the following files:


1) Martin_Harper_RunAnalysis.py - a python script that implements the modules contained in the following two scripts: 

2) FDP_corpus_PY3.py - a python script that creates the corpus from plain text files, contains the the top word and top document functions, and contains the functions used to select the terms used in the supervised PCA

3) FDP_SupPCA_PY3.py - a python script that creates the regression models, contains the classes used for cluster, ols, and fixed effects models. 

4) City848_Impartial_All_forPython_revised.csv - a csv files that contains file names and covariates for predictive models

5) city names.csv - a csv files that contains the names of every city in the analysis to be removed from the vocabulary list in addition to the stop words list

6) englishplus - a csv file containing the first set of words removed when cleaning the texts; this list is adapted from the "english" stop words list provided by the nltk package 

7) CITY_DATA - the directory of cleaned impartial analysis texts converted to plain text files 



Make sure the following packages are installed prior to running the Martin_Harper_RunAnalysis.py file:


csv
numpy
pandas
math
matplotlib
nltk
re
seaboarn
sklearn
statsmodels



When using these data, please use the following citation:

Martin, Isaac and Heather Harper. 2021. https://github.com/heatharper/Fiscal-Democracy-Project.git 

Martin, Isaac and Heather Harper. 2021. "What makes a tax policy popular? Predicting referendum votes from policy text." Socius. 









