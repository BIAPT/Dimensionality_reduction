import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA

data=pd.read_pickle('data/features/WSAS_wpli_10_01_alpha.pickle')
data_w=pd.read_pickle('data/features/WSAS_wpli_10_01_alpha.pickle')
data_d=pd.read_pickle('data/features/WSAS_dpli_10_01_alpha.pickle')
areas=data_w.columns[4:]

X =pd.DataFrame(np.hstack((data_d[areas],data_w[areas])))
X = X.iloc[np.where(data['Phase'] != 'Reco')[0],:]

data = data.iloc[np.where(data['Phase'] != 'Reco')[0],:]

#X = data[areas]
#X.shape

lab = np.zeros(len(data['Phase']))
lab[np.where(data['Phase']=='Anes')]=1
lab[np.where(data['Phase']=='Reco')]=2

plt.plot(lab)
plt.show()

participants=np.unique(data['ID'])


pca = PCA(n_components=3)
X_3 = pca.fit_transform(X)

pca = PCA(n_components=2)
X_2 = pca.fit_transform(X)

pdf = matplotlib.backends.backend_pdf.PdfPages("PCA_BA_path_wPLIdPLI.pdf")
phases = ['Base','Anes']

for part in participants:
    """
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    #ax.scatter(X_3[:, 0], X_3[:,1],X_3[:,2],marker='o', c=lab,s=40, alpha=0.2)
    n = np.where(data['ID'] == part)
    ax.scatter(X_3[n, 0], X_3[n,1],X_3[n,2], marker='o', c=lab[n],s=60)
    ax.plot(X_3[n, 0][0], X_3[n, 1][0],X_3[n,2][0],linewidth=0.4,color='k')
    ax.set_title(part)
    plt.show()
    pdf.savefig()
    plt.close()
    """

    fig = plt.figure(figsize=(10,10))
    scatter = plt.scatter(X_2[:, 0], X_2[:,1],marker='o', c=lab, alpha=0.2)
    plt.scatter(X_2[:, 0], X_2[:,1],marker='o', c=lab, alpha=0.2)
    plt.legend(handles=scatter.legend_elements()[0], labels=phases)
    n = np.where(data['ID'] == part)
    plt.scatter(X_2[n, 0], X_2[n, 1], marker='o', c=lab[n])
    plt.plot(X_2[n, 0][0], X_2[n, 1][0],linewidth=0.2,color='k')
    plt.title(part)
    plt.show()
    pdf.savefig()
    plt.close()

pdf.close()


