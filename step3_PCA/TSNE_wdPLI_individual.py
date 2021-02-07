import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
from sklearn.manifold import TSNE
from statsmodels.stats.multicomp import MultiComparison
import utils.stats as stats
from scikit_posthocs import posthoc_dunn

data_w=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('wPLI'))
#data_w=data_w[data_w['ID'] != "003MG"]
data_d=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('dPLI'))
#data_d=data_d[data_d['ID'] != "003MG"]

areas=data_w.columns[4:]
data = data_w.iloc[:,:4]
#X=np.array(data_d[areas])
X = np.hstack((data_w[areas],data_d[areas]))


label = np.zeros(len(data['Phase']))
label[np.where(data['Phase']=='Anes')]=1
label[np.where(data['Phase']=='Sedon1')]=1
label[np.where(data['Phase']=='emergencefirst5')]=1

participants = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
         'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22',
         'MCD0004', 'MCD0007', 'MCD0008', 'MCD0009', 'MCD0012', 'MCD0013','MCD0014', 'MCD0018', 'MCD0021',
         '002MG','003MG', '004MG', '004MW']

# 1: Recovered. 0: non-recovered 2: healthy  3:NET_ICU_Recovered
outcome = ['0', '1', '0', '1', '1', '1', '1', '1', '0', '0', '1',
         '2', '2', '2', '2', '2', '2','2','2','2',
         '3', '3', '3', '3']

oneD = []
Dist = []
meanA =[]
meanB =[]
pdf = matplotlib.backends.backend_pdf.PdfPages("TSNE_individual_PCA_BA_dwpli.pdf")

for part in participants:
    if part.__contains__('MW') or part.__contains__('MG'):
        phases = ['Sedoff', 'Sedon1']
    if part.__contains__('WSAS'):
        phases = ['Base', 'Anes']
    if part.__contains__('MCD'):
        phases = ['eyesclosed1', 'emergencefirst5']

    label_part = label[data['ID'] == part]
    X_part = X[data['ID'] == part].copy()
    # drop columns with nan
    X_part = X_part[:, ~np.isnan(X_part).any(axis=0)]

    X_2 = TSNE(n_components=2).fit_transform(X_part)

    X_1 = TSNE(n_components=1).fit_transform(X_part)

    data_one=pd.DataFrame(np.transpose(np.vstack((X_1.flatten(),label_part))))
    data_one.columns = ['conn_1D','label']
    oneD.append(data_one)

    mean_base = np.mean(X_1[label_part==0])
    mean_anes = np.mean(X_1[label_part==1])
    dist_tmp = max(mean_base,mean_anes)-min(mean_base,mean_anes)
    Dist.append(dist_tmp)
    meanA.append(mean_anes)
    meanB.append(mean_base)


    fig = plt.figure(figsize=(10, 10))
    scatter = plt.scatter(X_2[:, 0], X_2[:, 1], marker='o', c=label_part)
    plt.scatter(X_2[:, 0], X_2[:, 1], marker='o', c=label_part)
    plt.legend(handles=scatter.legend_elements()[0], labels=phases)
    plt.title(part)
    #plt.show()
    pdf.savefig()
    plt.close()

for i,p in enumerate(participants):
    data = oneD[i]
    mean_anes = meanA[i]
    mean_base = meanB[i]
    a = sns.displot(data=data, x = "conn_1D", hue = "label", kind = "kde",fill=True)
    a._legend.set_title(p)
    plt.axvline(x=mean_base, c = 'blue')
    plt.axvline(x=mean_anes, c = 'orange')
    pdf.savefig()


toplot = pd.DataFrame(np.transpose(np.vstack((Dist,outcome))),columns=["Distance","outcome"])
toplot = toplot.astype(float)

R = toplot.query("outcome==0")['Distance']
N = toplot.query("outcome==1")['Distance']
H = toplot.query("outcome==2")['Distance']

fvalue, pvalue, test = stats.ANOVA_assumptions_test(R, N, H)

plt.figure()
sns.boxplot(x = 'outcome', y = 'Distance', data = toplot)
sns.stripplot(x = 'outcome', y = 'Distance', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['non-recovered','recoveded','healthy','NET_ICU'])
plt.title("Distance \n {} f-value {}, p-value {}\n".format(str(fvalue)[0:5], test, str(pvalue)[0:5]))
pdf.savefig()

toplot = toplot.query("outcome != 3")
# POSTHOC TEST
if test == 'kruskal':
    toprint = posthoc_dunn(toplot, val_col='occurence', group_col='group', p_adjust='bonferroni')
    title = "DUNN Test"

if test == 'ANOVA':
    # perform multiple pairwise comparison (Tukey's HSD)
    MultiComp = MultiComparison(toplot['Distance'],
                                toplot['outcome'])
    toprint = pd.DataFrame(MultiComp.tukeyhsd().summary())
    title = "TUKEY Test"

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=toprint.values, colLabels=toprint.columns, loc='center')
plt.title('{} clutser'.format(title))
pdf.savefig(fig, bbox_inches='tight')


pdf.close()
