import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA

mode = "wpli"
data=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format(mode))
data=data[data['ID']!="003MG"]
areas=data.columns[4:]
X = data[areas]
X.shape

label = np.zeros(len(data['Phase']))
label[np.where(data['Phase']=='Anes')]=1
label[np.where(data['Phase']=='Sedoff')]=2
label[np.where(data['Phase']=='Sedon1')]=3
label[np.where(data['Phase']=='eyesclosed1')]=4
label[np.where(data['Phase']=='emergencefirst5')]=5
#plt.plot(label)
#plt.show()
phases = ['Base', 'Anes','Sedoff', 'Sedon1','eyesclosed1', 'emergencefirst5']

participants = np.unique(data['ID'])
len(participants)

pdf = matplotlib.backends.backend_pdf.PdfPages("Combined_PCA_BA_path_{}.pdf".format(mode))

pca = PCA(n_components=2)
X_2 = pca.fit_transform(X)

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

pdf.close()



