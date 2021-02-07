import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


data_w=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('wPLI'))
data_w=data_w[data_w['ID'] != "003MG"]
data_w=data_w[data_w['Phase'] != "Sedon1"]
data_w=data_w[data_w['Phase'] != "Sedoff"]
data_w=data_w[data_w['Phase'] != "emergencefirst5"]
data_w=data_w[data_w['Phase'] != "eyesclosed1"]
data_d=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('dPLI'))
data_d=data_d[data_d['ID'] != "003MG"]
data_d=data_d[data_d['Phase'] != "emergencefirst5"]
data_d=data_d[data_d['Phase'] != "eyesclosed1"]
data_d=data_d[data_d['Phase'] != "Sedon1"]
data_d=data_d[data_d['Phase'] != "Sedoff"]

areas=data_w.columns[4:]
data = data_w.iloc[:,:4]
X = np.hstack((data_w[areas],data_d[areas]))

label = np.zeros(len(data['Phase']))
label[np.where(data['Phase']=='Anes')]=1
#label[np.where(data['Phase']=='Sedoff')]=2
#label[np.where(data['Phase']=='Sedon1')]=3
#label[np.where(data['Phase']=='eyesclosed1')]=4
#label[np.where(data['Phase']=='emergencefirst5')]=5
#plt.plot(label)
#plt.show()
#phases = ['Base', 'Anes','Sedoff', 'Sedon1','eyesclosed1', 'emergencefirst5']
#phases = ['Base', 'Anes','Sedoff', 'Sedon1']
phases = ['Base', 'Anes']

participants = np.unique(data['ID'])
len(participants)

X_2 = TSNE(n_components=2).fit_transform(X)
X_2P = PCA(n_components=2).fit_transform(X)

pdf = matplotlib.backends.backend_pdf.PdfPages("TSNE_WSAS_group_PCA_BA_wdpli.pdf")

for part in participants:
    fig = plt.figure(figsize=(10,10))
    scatter = plt.scatter(X_2[:, 0], X_2[:,1],marker='o', c=label, alpha=0.2)
    plt.scatter(X_2[:, 0], X_2[:,1],marker='o', c=label, alpha=0.2)
    plt.legend(handles=scatter.legend_elements()[0], labels=phases)
    n = np.where(data['ID'] == part)
    plt.scatter(X_2[n, 0], X_2[n, 1], marker='o', c=label[n])
    plt.plot(X_2[n, 0][0], X_2[n, 1][0],linewidth=0.2,color='k')
    plt.title(part)
    plt.show()
    pdf.savefig()
    plt.close()

    fig = plt.figure(figsize=(10,10))
    scatter = plt.scatter(X_2P[:, 0], X_2P[:,1],marker='o', c=label, alpha=0.2)
    plt.scatter(X_2P[:, 0], X_2P[:,1],marker='o', c=label, alpha=0.2)
    plt.legend(handles=scatter.legend_elements()[0], labels=phases)
    n = np.where(data['ID'] == part)
    plt.scatter(X_2P[n, 0], X_2P[n, 1], marker='o', c=label[n])
    plt.plot(X_2P[n, 0][0], X_2P[n, 1][0],linewidth=0.2,color='k')
    plt.title(part)
    plt.show()
    pdf.savefig()
    plt.close()

pdf.close()




