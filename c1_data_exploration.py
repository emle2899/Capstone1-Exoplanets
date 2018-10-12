import pandas as pd
import numpy as np

# for stats
import scipy.stats as stats
from sklearn import preprocessing

# for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 18.0,
    'axes.titlesize'      : 'medium',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'medium',
})

# EDA
df_kep = pd.read_csv('data/exoplanet.csv')


# assign false positive to 0 and candidate to 1
change = {'FALSE POSITIVE':0,'CANDIDATE':1}
df_kep = df_kep.replace({'koi_pdisposition':change})

# Plot
def pie_chart(sizes,labels):
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.savefig('pie_chart')
    plt.show()

def plot_hist(df,title,labels = None):

    ax = df.hist(bins=15, normed=1, figsize=(10,8), alpha = .8, label = labels)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    plt.xlim([0, .25])
    plt.legend()
    plt.savefig('exoplanet_star_sizes')
    plt.show()

def plot_scatter(x,y,size,title,xlabel,ylabel,scale = 1e3):
    # scale sizes
    s = size*scale
    plt.scatter(x, y, c = 'y', s = s, alpha = 0.6)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('scatter')
    plt.show()


if __name__ == "__main__":
    confirmed = df_kep.query('koi_disposition == "CONFIRMED"')
    candidate = df_kep.query('koi_pdisposition == 1')
    false_pos = df_kep.query('koi_pdisposition == 0')

    # Pie chart of candidates and false positives
    labels = 'Candidate', 'False Positive'
    per_false_pos = len(false_pos)/len(df_kep.koi_pdisposition)
    per_candidate = len(candidate)/len(df_kep.koi_pdisposition)
    sizes = [per_false_pos,per_candidate]
    pie_chart(sizes,labels)

    # histogram of frequency of relative exoplanet sizes
    plot_hist(confirmed.koi_ror,'Freq of Relative Planet Sizes','confirmed')

    title, xlabel, ylabel = ['Depth VS Dist','Transit Depth','Distance from Star']
    x, y = [confirmed.koi_depth, confirmed.koi_dor]
    size = confirmed.koi_ror
    plot_scatter(x,y,size,title,xlabel,ylabel)
