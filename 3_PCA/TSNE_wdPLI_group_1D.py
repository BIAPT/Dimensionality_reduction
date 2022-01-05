import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
from sklearn.manifold import TSNE


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

outcome = np.zeros(len(data['ID']))
outcome[np.where(data['ID']=='WSAS02')]=10
outcome[np.where(data['ID']=='WSAS09')]=10
outcome[np.where(data['ID']=='WSAS19')]=10
outcome[np.where(data['ID']=='WSAS20')]=10


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

X_1 = TSNE(n_components=1).fit_transform(X)
data_one = pd.DataFrame(np.transpose(np.vstack((X_1.flatten(), label,outcome))))
data_one.columns = ['conn_1D', 'label','outcome']
data_one.index=data.index


ax = sns.displot(data=data_one, x="conn_1D", hue="label", kind="kde", col="outcome", fill=True)
ax.set_titles("{col_name} participants")
plt.show()





