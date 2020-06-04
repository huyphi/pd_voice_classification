"""
Huy Phi
CSE 163 Section AA

This file implements the functions for my CSE 163 project.
Specifically, these functions generate the machine learning models,
and assess these models by using confusion matrix and ROC curves.
It also generates figures, including barchart demographics,
box plots of significant features, heatmap confusion matrix,
and the actual ROC curves. Designed to be used as a module for main.py.
"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def plot_demographics(data):
    """
    Takes in a pandas dataframe and generates a barchart demonstrating
    the patient gender distribution among healthy and PD groups.
    Saves the barchart as a file called pd_demographics.png
    """
    sns.set()

    # Data prep
    data = data.copy()
    # Replaces the binary indicator with PD/Healthy for plot labeling
    data["class"].replace({0: "Healthy", 1: "PD"}, inplace=True)
    data["gender"].replace({0: "Female", 1: "Male"}, inplace=True)
    # Drops duplicate ids to get the true patient count
    data.drop_duplicates(subset="id", inplace=True)
    class_id = data[["class", "id", "gender"]]

    # plot the barplot
    sns.catplot(data=class_id, x="class", kind="count", hue="gender",
                palette="Blues")
    plt.xlabel("Diagnosis")
    plt.title("PD Distribution in the Dataset")
    plt.savefig("pd_demographics.png", dpi=1000, bbox_inches="tight")
    plt.clf()


def plot_box_plot(data):
    """
    Takes in the dataframe and plots box plots for the top 4 features with
    significant differences among PD and healthy participants in the current
    dataset. All features have been calculated previously, and have been
    hardcoded into this function. Saves the figure as boxplot.png.
    """
    data = data.copy()
    # Replaces the binary indicator with PD/Healthy for plot labeling
    data["class"].replace({0: "Healthy", 1: "PD"}, inplace=True)

    # Generates plot. Order and showfliers was set for aesthetic purposes.
    fig, ax = plt.subplots(2, 2)
    sns.boxplot(x="class", y='mean_MFCC_2nd_coef', data=data, showfliers=False,
                palette="coolwarm", ax=ax[0, 0], order=["Healthy", "PD"])

    sns.boxplot(x="class", y='tqwt_minValue_dec_12', data=data,
                showfliers=False, palette="coolwarm", ax=ax[0, 1],
                order=["Healthy", "PD"])

    sns.boxplot(x="class", y='tqwt_stdValue_dec_12', data=data,
                showfliers=False, palette="coolwarm", ax=ax[1, 0],
                order=["Healthy", "PD"])

    sns.boxplot(x="class", y='tqwt_maxValue_dec_12', data=data,
                showfliers=False, palette="coolwarm", ax=ax[1, 1],
                order=["Healthy", "PD"])

    # generalizes the x and y axes for each subplot
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlabel(" ")
            ax[i, j].set_ylabel("Feature Measure")

    # add the title for each feature in the subplot
    ax[0, 0].set_title("Mean MFCC 2nd Coefficient")
    ax[0, 1].set_title("TQWT 12th Decibel Min Value")
    ax[1, 0].set_title("TQWT 12th Decibel Standard Deviation")
    ax[1, 1].set_title("TQWT 12th Decibel Max Value")

    plt.tight_layout()
    plt.savefig("boxplot.png", dpi=1000, bbox_incehs="tight")
    plt.clf()


def plot_confusion_matrix(models, features_test, labels_test):
    """
    Takes in a list of fitted machine learning models, the test set
    features and labels, and plots confusion matrix heatmaps
    for each model. Saves the plot as confusion_matrix.png
    """
    # Sets up plot
    fig, ax = plt.subplots(2, 2)
    i = 0
    for ax_row in range(2):
        for ax_col in range(2):
            labels_pred = models[i].predict(features_test)
            # generates confusion matrix heatmap for each model
            data = confusion_matrix(labels_test, labels_pred)
            sns.heatmap(data=data, ax=ax[ax_row][ax_col], annot=True, fmt='g',
                        xticklabels=['Pred -', 'Pred +'], cmap="Blues",
                        yticklabels=['True -', 'True +'], square=True)
            model_label = str(models[i]).split("(")
            ax[ax_row][ax_col].set_title(model_label[0])
            i += 1
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=1000, bbox_inches="tight")
    plt.clf()


def cross_val_assessment(model, features, labels, n):
    """
    Takes in a model, features, labels and an integer n, and generates
    cross-validated confusion matrix measures, including recall, precision,
    F1-score and accuracy. Prints out the mean and standard deviation for each
    measure based on n folds via k-folds method of cross-validation.
    """
    print("Confusion Matrix:")
    # Prints out mean and sd for each score
    scores = cross_val_score(model, features, labels, cv=n, scoring="recall")
    print("Cross Validated Recall: %0.2f (+/- %0.2f)" %
          (scores.mean(), scores.std()))
    scores = cross_val_score(model, features, labels, cv=n,
                             scoring="precision")
    print("Cross Validated Precision: %0.2f (+/- %0.2f)" %
          (scores.mean(), scores.std()))
    scores = cross_val_score(model, features, labels, cv=n, scoring="f1")
    print("Cross Validated F1 Score: %0.2f (+/- %0.2f)" %
          (scores.mean(), scores.std()))
    scores = cross_val_score(model, features, labels, cv=n, scoring="accuracy")
    print("Cross Validated Accuracy: %0.2f (+/- %0.2f)" %
          (scores.mean(), scores.std()))
    print("")


def roc_plot_all(models, features, labels):
    """
    Takes in a list of fitted machine learning models and the
    features/labels in a dataset, and generates ROC curves for each
    model. Generates ROC curve plot in a file called roc.png, and prints out
    the AUC for each model.
    """
    print("Generating ROC Curves...")
    print("")
    # generate plots for no-skill
    ns_probs = [0 for _ in range(len(labels))]
    ns_auc = roc_auc_score(labels, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(labels, ns_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle="--",
             label="No Skill: AUC = %.3f" % (ns_auc))

    # roc curve for each model and generates
    for model in models:
        # calculate model probability and AUC
        model_probs = model.predict_proba(features)
        model_probs = model_probs[:, 1]
        model_auc = roc_auc_score(labels, model_probs)
        model_label = str(model).split("(")
        print(model_label[0] + " ROC: AUC = %.3f" % (model_auc))
        # plot fpr vs. tpr to generate plot
        fpr, tpr, _ = roc_curve(labels, model_probs)
        plt.plot(fpr, tpr,
                 label=model_label[0] + ": AUC = %.3f" % (model_auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for PD Classification Models")
    plt.legend(prop={'size': 10})
    plt.savefig('roc.png', dpi=1000, bbox_inches="tight")


def classifier_comparison(data, n):
    """
    Takes in a PD dataframe of the data, splits the data 80% test
    and 20% train and generates machine learning classifier
    models, including decision tree, random forest, logistic
    regression, and K nearest neighbors.
    Plugs these models into confusion matrix function, and the
    ROC curve function defined previously.
    Assumes the datas binary target is the "class".
    n is th e number of K-folds that want to be used in cross validation.
    """
    # data prep into features, labels and 80:20 train/test split
    ml_data = data.loc[:, data.columns != "id"]
    features = ml_data.loc[:, ml_data.columns != "class"]
    labels = ml_data["class"]
    # random_state is set for reproducability
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.20, random_state=2)

    # building models
    print("Building, fitting, anc cross validating models...")
    print("")
    dt_model = DecisionTreeClassifier()
    rf_model = RandomForestClassifier(n_estimators=50)
    lr_model = LogisticRegression(solver="newton-cg", penalty="none",
                                  max_iter=50)
    knn_model = KNeighborsClassifier()

    # model fitting
    models = [dt_model, rf_model, lr_model, knn_model]
    for model in models:
        # fit each model to the same training data
        model.fit(features_train, labels_train)
        print(str(model))
        # generate confusion matrix cross-validated measures for each model
        cross_val_assessment(model, features, labels, n)

    # Plotting the confusion matrix
    plot_confusion_matrix(models, features_test, labels_test)

    # Generate ROC for each model
    roc_plot_all(models, features, labels)
