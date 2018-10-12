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
    plt.tight_layout()
    # plt.savefig('chart')
    plt.show()

def plot_hist(df,title,labels = None):

    ax = df.hist(bins=15, normed=1, figsize=(10,8), alpha = .8, label = labels)
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    plt.xlim([0, .25])
    plt.legend()
    plt.tight_layout()
    # plt.savefig('exoplanet_hist')
    plt.show()

def plot_scatter(x,y,size,title,xlabel,ylabel, labels, color = 'y', scale = 1e3):
    # scale sizes
    s = size*scale
    plt.scatter(x, y, c = color, s = s, alpha = 0.6,label = labels)
    plt.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend(loc=1, prop={'size': 6})
    plt.savefig('scatter_plot_colorlegend')
    plt.show()

def color(arr, range):
    colors = []
    for i in arr:
        if i < range[0]:
            colors.append('red')
        elif i < range[1]:
            colors.append('orange')
        elif i < range[2]:
            colors.append('pink')
        elif i < range[3]:
            colors.append('blue')
        else:
            colors.append('purple')
    return colors

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
    range = [0.1,0.2,0.3,0.4]
    c = color(size, range)
    labels2 = np.array([['red < 0.1'],['orange <0.2'],['pink <0.3'],['blue<0.4'],['purple >0.4']])
    plot_scatter(x,y,size,title,xlabel,ylabel, color = c, labels = labels2)
