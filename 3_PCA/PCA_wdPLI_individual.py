import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA
import seaborn

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

participants = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
         'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22',
         'MCD0004', 'MCD0007', 'MCD0008', 'MCD0009', 'MCD0012', 'MCD0013','MCD0014', 'MCD0018', 'MCD0021',
         '002MG', '004MG', '004MW']

pdf = matplotlib.backends.backend_pdf.PdfPages("individual_PCA_BA_wdpli.pdf")

for part in participants:
    if part.__contains__('MW') or part.__contains__('MG'):
        phases = ['Sedoff', 'Sedon1']
    if part.__contains__('WSAS'):
        phases = ['Base', 'Anes']
    if part.__contains__('MCD'):
        phases = ['eyesclosed1', 'emergencefirst5']

    X_part = X[data['ID'] == part]
    label_part = label[data['ID'] == part]

    pca = PCA(n_components=2)
    X_part.shape
    X_part_2 = pca.fit_transform(X_part.astype(np.float))

    fig = plt.figure(figsize=(10,10))
    scatter = plt.scatter(X_part_2[:, 0], X_part_2[:,1],marker='o', c=label_part)
    plt.scatter(X_part_2[:, 0], X_part_2[:,1],marker='o', c=label_part)
    plt.legend(handles=scatter.legend_elements()[0], labels=phases)
    plt.title(part)
    #plt.show()
    pdf.savefig()
    plt.close()

    pca = PCA(n_components=3)
    X_part_3 = pca.fit_transform(X_part.astype(np.float))

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    #scatter= plt.scatter(X_part_3[:, 0], X_part_3[:,1],X_part_3[:,2], marker='o', c=label_part)
    ax.scatter(X_part_3[:, 0], X_part_3[:,1],X_part_3[:,2], marker='o', c=label_part,s=60)
    ax.set_title(part)
    #plt.show()
    pdf.savefig()
    plt.close()


pdf.close()




