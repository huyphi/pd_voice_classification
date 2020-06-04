"""
Huy Phi
CSE 163 Section AA:
Final Project

This file is the Main file for my CSE 163 final project, in diagnosing
Parkinsons Disease using voice. This file reads in the data, and
has functions to generate p-values for each feature, and drops
non-significant features from the data that are above a p-value
threshold. Then calls the many functions in module functions.py.
This file is the module that runs the entire project.
"""
from functions import plot_demographics, plot_box_plot, classifier_comparison
import pandas as pd
from scipy.stats import ttest_ind
import operator
import warnings
from sklearn.exceptions import ConvergenceWarning


def gen_p_values(data):
    """
    Takes in a pandas dataframe and returns a dictionary with each column
    feature mapped to the p-value generated from independent t-test
    of means between a binary class, I.E. PD vs healthy.
    """
    # filters the data into binary classes
    healthy_data = data[data["class"] == 0]
    pd_data = data[data['class'] == 1]
    p_values = {}
    for feature in healthy_data.columns:
        # excludes class, gender and id
        if feature == 'class' or feature == 'gender' or feature == 'id':
            p_values[feature] = 0
        else:
            # calculates p value for independent t test for PD vs Healthy
            cur_healthy = healthy_data[feature]
            cur_pd = pd_data[feature]
            ttest = ttest_ind(cur_healthy, cur_pd)
            p_values[feature] = ttest.pvalue

    return p_values


def find_sig_pvalues(p_values, thr):
    """
    Takes in a dictionary of features mapped to p-values, and a threshold
    cutoff for p-values and returns a new dictionary any features with
    p-values above the threshold removed from the input dictionary.
    """
    sig_pvalues = {}
    for metric, pvalue in p_values.items():
        if pvalue < thr:
            sig_pvalues[metric] = pvalue
    # adds in class and gender
    return sig_pvalues


def top_sig_features(sig_features, n):
    """
    Takes in a dictionary of the significant features mapped to their
    p-values and prints out the n number of features with the lowest
    p-values.
    """
    sig_features = sorted(sig_features.items(), key=operator.itemgetter(1))
    print("These are the " + str(n) + " features with the smallest p-values:")
    print("")
    for feature in sig_features[3: 3 + n]:
        print(feature)


def main():
    # ignores convergence warnings sometimes generated by fitting logistic reg
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    print("Beginning Analysis...")
    print("")
    data = pd.read_csv("pd_speech_features.csv", skiprows=1)

    # find significant p-values

    # pvalues for every column
    pvalue = gen_p_values(data)

    # pvalues deemed significant that are below the threshold.
    sig_features = find_sig_pvalues(pvalue, 0.05)   # Threshold can be adjusted
    non_sig_features = list(set(data.columns) - set(sig_features))
    # drops non-significant features from the dataset
    sig_data = data.drop(non_sig_features, axis=1)

    print("Number of Original Features: " + str(len(pvalue)))
    print("Number of Significant Features: " + str(len(sig_features)))
    print("")
    # Prints out the 10 features with the lowest p-values, n can be adjusted
    top_sig_features(sig_features, 10)
    print("")

    # generate barchart and boxplots
    plot_demographics(data)
    plot_box_plot(sig_data)
    # makes all the machine learning models, and generates 5 K-folds.
    classifier_comparison(sig_data, 5)   # Number of folds can be adjusted
    print("")
    print("Analysis Done! :^)")


if __name__ == "__main__":
    main()
