import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA

data_w=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('wPLI'))
data_w=data_w[data_w['ID']!="003MG"]
data_d=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('dPLI'))
data_d=data_d[data_d['ID']!="003MG"]

areas=data_w.columns[4:]
data = data_w.iloc[:,:4]
X = np.hstack((data_w[areas],data_d[areas]))

X.shape

label = np.zeros(len(data['Phase']))
label[np.where(data['Phase']=='Anes')]=1
label[np.where(data['Phase']=='Sedon1')]=1
label[np.where(data['Phase']=='emergencefirst5')]=1

#plt.plot(label)
#plt.show()

participants=np.unique(data['ID'])
len(participants)

pca = PCA(n_components=2)
X_2 = pca.fit_transform(X)

pdf = matplotlib.backends.backend_pdf.PdfPages("PCA_BA_path_wPLIdPLI.pdf")
phases = ['Base','Anes']

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


