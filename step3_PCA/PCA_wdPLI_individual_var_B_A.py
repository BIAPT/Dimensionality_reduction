import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA
import seaborn as sns
from entropy import entropy

data_w=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('wPLI'))
data_w=data_w[data_w['ID'] != "003MG"]
#data_d=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('dPLI'))
#data_d=data_d[data_d['ID'] != "003MG"]

areas=data_w.columns[4:]
data = data_w.iloc[:,:4]
X = data_w[areas]
#X = np.hstack((data_w[areas],data_d[areas]))

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

std_Base = []
std_Anes = []
entr_Base = []
entr_Anes = []

for part in participants:
    if part.__contains__('MW') or part.__contains__('MG'):
        phases = ['Sedoff', 'Sedon1']
    if part.__contains__('WSAS'):
        phases = ['Base', 'Anes']
    if part.__contains__('MCD'):
        phases = ['eyesclosed1', 'emergencefirst5']

    label_part = label[data['ID'] == part]
    X_part = data_w.query("ID =='{}'".format(part))
    X_part_B = X_part.query("Phase =='{}'".format(phases[0]))[areas]
    X_part_A = X_part.query("Phase =='{}'".format(phases[1]))[areas]

    std_Base.append(np.mean(np.std(X_part_B)))
    std_Anes.append(np.mean(np.std(X_part_A)))

    tmp_ent =[]
    for a in range(len(areas)):
        tmp_ent.append(entropy.perm_entropy(X_part_A[areas[a]], order=10, normalize=True))  # Permutation entropy
    entr_Anes.append(np.mean(tmp_ent))

    tmp_ent =[]
    for a in range(len(areas)):
        tmp_ent.append(entropy.perm_entropy(X_part_B[areas[a]], order=10, normalize=True))  # Permutation entropy
    entr_Base.append(np.mean(tmp_ent))

#pdf = matplotlib.backends.backend_pdf.PdfPages("individual_PCA_BA_wdpli_1D.pdf")

toplot = pd.DataFrame()
toplot['ent_B'] = entr_Base
toplot['ent_A'] = entr_Anes
toplot['std_B'] = std_Base
toplot['std_A'] = std_Anes
toplot['ID'] = participants
toplot['outcome'] = outcome

toplot_long = toplot.melt(id_vars=['outcome'],value_vars=['ent_B','ent_A'])
toplot_long.columns = ['outcome','phase','entr']

plt.figure()
sns.boxplot(x = 'outcome', y = 'entr',hue = 'phase', data = toplot_long)
sns.stripplot(x = 'outcome', y = 'entr',hue = 'phase', size=4, color=".3",data = toplot_long)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("Entropy")
plt.show()

toplot_long = toplot.melt(id_vars=['outcome'],value_vars=['std_B','std_A'])
toplot_long.columns = ['outcome','phase','std']

plt.figure()
sns.boxplot(x = 'outcome', y = 'std',hue = 'phase', data = toplot_long)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("PCA_1_explained_Variance")
plt.show()

difference_ent = toplot[['ent_A','ent_B']].max(axis=1)-toplot[['ent_A','ent_B']].min(axis=1)
difference = toplot[['std_A','std_B']].max(axis=1)-toplot[['std_A','std_B']].min(axis=1)

toplot['difference_ent'] = difference_ent
toplot['difference'] = difference

plt.figure()
sns.boxplot(x = 'outcome', y = 'difference', data = toplot)
sns.stripplot(x = 'outcome', y = 'difference', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("PCA_1_explained_Variance")
plt.show()

plt.figure()
sns.boxplot(x = 'outcome', y = 'difference_ent', data = toplot)
sns.stripplot(x = 'outcome', y = 'difference_ent', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("PCA_1_explained_Variance")
plt.show()

# libraries
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

# Make the plot
plt.figure()
parallel_coordinates(toplot[['ent_A','ent_B','outcome']], 'outcome', colormap=plt.get_cmap("Set2"))
plt.show()



#pdf.savefig()
#pdf.close()




