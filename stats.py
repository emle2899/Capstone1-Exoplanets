import pandas as pd
import numpy as np

# for stats
import scipy.stats as stats
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import linear_model
from sklearn import metrics

# for plotting and tables
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 18.0,
    'axes.titlesize'      : 'medium',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'medium',
})


def calculate_vif(X, thresh=10.0):
    '''
    Values with VIF > thresh are too collinear and will be removed from dataframe
    Standard thresh is 10
    INPUT:
    X = DataFrame
    thresh = float
    OUTPUT:
    dataframe
    '''
    dropped=True
    while dropped:
        col = X.columns
        dropped = False
        vif = [variance_inflation_factor(X[col].values, col.get_loc(var)) for var in col]

        max_vif = max(vif)
        if max_vif > thresh:
            value = vif.index(max_vif)
            print(f'Dropping {X.columns[value]} with vif={max_vif}')
            X = X.drop([X.columns.tolist()[value]], axis=1)
            dropped=True
    return X

def scale_data(X):
    '''
    Standardize and scale data
    INPUT:
    X = array
    OUTPUT:
    array
    '''
    x = X.values
    snd_scaler = preprocessing.StandardScaler()
    mm_scaler = preprocessing.MinMaxScaler()
    x_scaled = mm_scaler.fit_transform(x)
    X_scaled = pd.DataFrame(data=x_scaled,columns=X.columns, index=X.index)

    return X_scaled

def make_model(X_train, y_train):
    '''
    Makes a logistic regression model and returns probabilities, predictions, coefficients,
    and the classification and confusion matrices
    INPUT:
    X_train = dataframe
    y_train = dataframe
    OUTPUT:
    array, array, array, array, array, array
    '''
    model = linear_model.LogisticRegression()
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)

    # find coeffiecents, classification matrix, and confusion matrix
    coef = model.coef_
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    return probabilities, y_pred, coef, precision, recall, cnf_matrix

def roc(probabilities, labels):
    '''
    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    INPUT:
    probabilities = numpy array
    labels = numpy array
    OUTPUT:
    list, list, list
    '''

    thresholds = np.sort(probabilities)
    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        predicted_positive = probabilities >= threshold
        true_positives = np.sum(predicted_positive * labels)
        false_positives = np.sum(predicted_positive) - true_positives

        tpr = true_positives / float(num_positive_cases)
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()

def plot_roc(fpr,tpr,title):
    '''
    Plot ROC curve
    INPUT:
    fpr, tpr = array
    title = string
    OUTPUT:
    plt.plot
    '''
    plt.plot(fpr, tpr,label='auc = {0:.2f}'.format(metrics.auc(fpr,tpr)))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.savefig('plotting_roc_curve')
    plt.show()

# for plotting and tables:
def heatplot(X):
    '''
    Plot correlation heatmap
    INPUT:
    matrix = DataFrame
    OUTPUT:
    sns.heatmap
    '''
    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.zeros_like(X, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(X, mask=mask, cmap=cmap, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title('Pearson Correlation Coeficient Matrix')
    plt.savefig('correlation_plot')
    plt.show()

def to_dataframe(lst,col = None):
    '''
    convert list to dataframe
    INPUT:
    lst = list
    col = string
    OUTPUT:
    dataframe
    '''
    arr = np.array([lst])
    return pd.DataFrame(arr, columns=col)

def to_markdown(df, round_places=3):
    '''
    Returns a markdown, rounded representation of a dataframe
    INPUT:
    df = DataFrame
    OUTPUT:
    markdown table
    '''
    print(tabulate(df.round(round_places), headers='keys', tablefmt='pipe'))


if __name__ == "__main__":
    df = pd.read_csv('data/cleaned.csv')
    X = df.iloc[0:1000].drop('Unnamed: 0', axis=1)
    y = X.pop('koi_pdisposition')

    # splitting data
    X_scaled = scale_data(X)
    X= calculate_vif(X_scaled,thresh=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=130)
    model_table = to_markdown(X_train.iloc[0:10])

    # Compute pairwise correlation of remaining columns
    correlated = X_train.corr(method='pearson', min_periods=1)
    heatplot(correlated)

    # Make predictive model and plot roc curve
    probabilities, y_pred, coef, precision, recall, cnf_matrix = make_model(X_train, y_train)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probabilities)
    plot_roc(fpr,tpr,title='ROC plot of Model')

    # Put info into tables
    df_prec_recall = to_dataframe([precision, recall],col = ['precision', 'recall'])
    prec_table = to_markdown(df_prec_recall)
