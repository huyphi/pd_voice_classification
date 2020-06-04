# Diagnosing Parkinson's Disease using Voice:
## By Huy Phi:

Hi! My name is Huy and welcome to my project. I have keen interest in neurodegenerative diseases and decided to try to do some machine learning classification of Parkinson's Disease (PD) using voice extracted features. This project encapsulates the programming concepts I learned in CSE 163 and allowed me to apply to something I learned in the classroom, and in my research.


## Files

`pd_speech_features.csv` is the csv file with all the speech features extracted from 188 patients with PD and 64 healthy controls. Each participant provided 3 voice samples, and 754 columns of features were extracted from various speech signal processing algorithms. This is the file we will be analyzing!

`main.py` this is the "main" file, that will be used to filter out the data for feature selection using statistical methods, and incorporate the functions generated in other files to output the results.

`functions.py` is the file that implements the machine learning methods, including generating and fitting models, and evaluating them via confusion matrix and ROC curves.

`test.py` is the file where I implemented tests for the statistical methods used. This was mainly used to test if I calculated p-values correctly, and outputted what I wanted to output from these functions in `main.py`

`stat_test.csv` and `stat_test2.csv` are self-generated csv files that I used to test out functions in `test.py`

`cse_163_utils.py` is the CSE 163 provided utility file that implements functions for testing.

## Instructions

Simply run the `main.py` script with all the other neccessary files in the directory described above. This will print out values in the terminal, including a select number of features with the lowest p-values. It also assess machine learning models, and prints out values in the terminal including the precision, recall, F1-score, accuracy, and AUC for the models generated for diagnosing PD from voice samples. To run the testing files, simply run the `test.py` file.

To run this project, you will need the following libraries installed:

`pandas`, `seaborn`, `matplotlib.pyplot`, `sklearn`, `scipy.stats`, `operator` and `warnings`

The following functions in `main.py` have adjustable parameters:

`find_sig_pvalues(pvalue, n)` where n can be adjusted to designate any cutoff threshold for p-values. In `main.py` I set the threshold to be 0.05.

`top_sig_features(sig_features, n)` where n can be adjusted to designate the number of significant features you want to print out in a sorted order. In `main.py` I set it to print the top 10 features.

`classifier_comparison(sig_data, n)` is the function that implements all the machine learning. Here you can set n to be the number of K-folds you want to make in cross-validation. In `main.py` I set the number to 5.


## Files Generated

`pd_demographics.png` is a bar-chart showing the demographics of the dataset, including the gender distribution among the PD and Healthy groups.

`boxplot.png` is a box plot showing the top 4 features with the most statistically significant different voice features between PD and healthy groups.

`confusion_matrix.png` plots heatmaps of the confusion matrices for each model demonstrating performance on the test set.

`roc.png` plots ROC curves for each model, demonstrating general performance for binary classification models.