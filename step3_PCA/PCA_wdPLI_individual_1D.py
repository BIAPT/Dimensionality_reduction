import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA
import seaborn as sns

data_w=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('wPLI'))
data_w=data_w[data_w['ID'] != "003MG"]
data_d=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('dPLI'))
data_d=data_d[data_d['ID'] != "003MG"]

areas=data_w.columns[4:]
data = data_w.iloc[:,:4]
X=data_w[areas]
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

outcome = ['0', '1', '0', '1', '1', '1', '1', '1', '0', '0', '1',
         '2', '2', '2', '2', '2', '2','2','2','2',
         '3', '3', '3']
oneD = []
ex_var = []

for part in participants:
    if part.__contains__('MW') or part.__contains__('MG'):
        phases = ['Sedoff', 'Sedon1']
    if part.__contains__('WSAS'):
        phases = ['Base', 'Anes']
    if part.__contains__('MCD'):
        phases = ['eyesclosed1', 'emergencefirst5']

    label_part = label[data['ID'] == part]
    X_part = X[data['ID'] == part].copy()
    pca = PCA(n_components=1)
    data_one = pca.fit_transform(X_part)

    data_one=pd.DataFrame(np.transpose(np.vstack((data_one.flatten(),label_part))))
    data_one.columns = ['conn_1D','label']
    oneD.append(data_one)
    ex_var.append(pca.explained_variance_[0])


pdf = matplotlib.backends.backend_pdf.PdfPages("individual_PCA_BA_wdpli_1D.pdf")

toplot = pd.DataFrame()
toplot['variance'] = ex_var
toplot['ID'] = participants
toplot['outcome'] = outcome

plt.figure()
sns.boxplot(x = 'outcome', y = 'variance', data = toplot)
sns.stripplot(x = 'outcome', y = 'variance', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['non-recovered','recoveded','healthy','NET_ICU'])
plt.title("PCA_1_explained_Variance")
plt.show()
pdf.savefig()

for i,p in enumerate(participants):
    data = oneD[i]
    a = sns.displot(data=data, x = "conn_1D", hue = "label", kind = "kde",fill=True)
    a._legend.set_title(p)
    pdf.savefig()


pdf.close()




