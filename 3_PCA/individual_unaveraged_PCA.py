import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA
from scipy.io import loadmat
import seaborn as sns
import utils.stats as stats
from statsmodels.stats.multicomp import MultiComparison
from scikit_posthocs import posthoc_dunn
from entropy import entropy
from pandas.plotting import parallel_coordinates


mode = "wpli"
frequency = "alpha"
step = "01"
CONDITION = ["Base", "Anes"]

P_IDS = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
         'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22',
         'MCD0004', 'MCD0007', 'MCD0008', 'MCD0009', 'MCD0012', 'MCD0013','MCD0014', 'MCD0018', 'MCD0021',
         '002MG', '003MG', '004MG', '004MW']
outcome = ['0', '1', '0', '1', '1', '1', '1', '1', '0', '0', '1',
         '2', '2', '2', '2', '2', '2','2','2','2',
         '3', '3', '3', '3']

INPUT_DIR = "data/connectivity/{}/{}/step{}/".format(frequency, mode,step)

oneD = []
Dist = []
meanA =[]
meanB =[]
std_Base = []
std_Anes = []
exp_Base = []
exp_Anes = []
entr_Base = []
entr_Anes = []

pdf = matplotlib.backends.backend_pdf.PdfPages("individual_unaveraged_PCA_BA_{}.pdf".format(mode))

for p_id in P_IDS:
    # change the condition name for other groups
    if p_id.__contains__('MW') or p_id.__contains__('MG'):
        part_in = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(mode[0], frequency, step, p_id, 'sedoff')
        data_Base = loadmat(part_in)
        part_in = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(mode[0], frequency, step, p_id, 'sedon1')
        data_Anes = loadmat(part_in)
        part_channels = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(mode[0], frequency, step, p_id, 'sedoff')
        channel_Base = loadmat(part_channels)['channels'][0][0]
        part_channels = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(mode[0], frequency, step, p_id, 'sedon1')
        channel_Anes = loadmat(part_channels)['channels'][0][0]

    if p_id.__contains__('MCD'):
        part_in = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(mode[0], frequency, step, p_id, 'eyesclosed1')
        data_Base = loadmat(part_in)
        part_in = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(mode[0], frequency, step, p_id, 'emergencefirst5')
        data_Anes = loadmat(part_in)
        part_channels = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(mode[0], frequency, step, p_id, 'eyesclosed1')
        channel_Base = loadmat(part_channels)['channels'][0][0]
        part_channels = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(mode[0], frequency, step, p_id, 'emergencefirst5')
        channel_Anes = loadmat(part_channels)['channels'][0][0]

    if p_id.__contains__('WSAS'):
        part_in = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(mode[0], frequency, step, p_id, 'Base')
        data_Base = loadmat(part_in)
        part_in = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(mode[0], frequency, step, p_id, 'Anes')
        data_Anes = loadmat(part_in)
        part_channels = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(mode[0], frequency, step, p_id, 'Base')
        channel_Base = loadmat(part_channels)['channels'][0][0]
        part_channels = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(mode[0], frequency, step, p_id, 'Anes')
        channel_Anes = loadmat(part_channels)['channels'][0][0]

    # select data
    data_Base = data_Base["{}pli_tofill".format(mode[0])]
    data_Anes = data_Anes["{}pli_tofill".format(mode[0])]
    print('Load data comlpete {}'.format(p_id))
    electrodes = len(channel_Base)

    if len(channel_Base) != len(channel_Anes):
        inboth = np.intersect1d(channel_Base,channel_Anes)
        select = np.where(np.isin(channel_Base,inboth))[0]
        data_Base = data_Base[:,select,:]
        data_Base = data_Base[:,:, select]

        select = np.where(np.isin(channel_Anes,inboth))[0]
        data_Anes = data_Anes[:,select,:]
        data_Anes = data_Anes[:,:, select]
        electrodes = len(select)

    # reshape the data
    data_Base = data_Base.transpose(2,1,0)
    data_Anes = data_Anes.transpose(2,1,0)

    #EARLIER IT WAS THIS: THIS IS AN ERROR:
    #data_Anes.reshape(electrodes,electrodes,data_Anes.shape[0])

    # transform to long data
    data_Base_long = data_Base[np.triu_indices(electrodes-1)]
    data_Anes_long = data_Anes[np.triu_indices(electrodes-1)]

    std_Base.append(np.mean(np.std(data_Base_long,axis=1)))
    std_Anes.append(np.mean(np.std(data_Anes_long,axis=1)))

    pca = PCA(n_components=3)
    pca.fit(np.transpose(data_Base_long))
    exp_Base.append(sum(pca.explained_variance_))

    pca = PCA(n_components=3)
    pca.fit(np.transpose(data_Anes_long))
    exp_Anes.append(sum(pca.explained_variance_))

    tmp_ent = []
    for a in range(len(select)):
        tmp_ent.append(entropy.perm_entropy(data_Anes_long[a], order=3, normalize=True))  # Permutation entropy
    entr_Anes.append(np.mean(tmp_ent))

    tmp_ent = []
    for a in range(len(select)):
        tmp_ent.append(entropy.perm_entropy(data_Base_long[a], order=3, normalize=True))  # Permutation entropy
    entr_Base.append(np.mean(tmp_ent))

    data = np.hstack((data_Base_long,data_Anes_long))
    label = np.ones(data.shape[1])
    label[0:data_Base_long.shape[1]] = 0
    phases = ['Base','Anes']

    pca = PCA(n_components=2)
    data_2 = pca.fit_transform(np.transpose(data))

    fig = plt.figure(figsize=(10,10))
    scatter = plt.scatter(data_2[:, 0], data_2[:,1],marker='o', c=label)
    plt.scatter(data_2[:, 0], data_2[:,1],marker='o', c=label)
    plt.legend(handles=scatter.legend_elements()[0], labels=phases)
    plt.title(p_id)
    pdf.savefig()
    plt.close()

    pca = PCA(n_components=3)
    data_3 = pca.fit_transform(np.transpose(data))

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    scatter = ax.scatter(data_3[:, 0], data_3[:, 1],data_3[:, 2], marker='o', c=label)
    ax.scatter(data_3[:, 0], data_3[:, 1],data_3[:, 2], marker='o', c=label)
    ax.plot(data_3[:, 0], data_3[:, 1],data_3[:, 2],linewidth=0.6,color='k')
    ax.set_title(p_id)
    pdf.savefig()
    plt.close()

    pca = PCA(n_components=1)
    data_1 = pca.fit_transform(np.transpose(data))

    data_one = pd.DataFrame(np.transpose(np.vstack((data_1.flatten(), label))))
    data_one.columns = ['conn_1D', 'label']
    oneD.append(data_one)

    mean_base = np.mean(data_1[label == 0])
    mean_anes = np.mean(data_1[label == 1])
    dist_tmp = max(mean_base, mean_anes) - min(mean_base, mean_anes)
    Dist.append(dist_tmp)
    meanA.append(mean_anes)
    meanB.append(mean_base)



for i,p in enumerate(P_IDS):
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
plt.xticks([0,1,2,3],['recovered','nonrecoveded','healthy','NET_ICU'])
plt.title("Distance \n {} f-value {}, p-value {}\n".format(str(fvalue)[0:5], test, str(pvalue)[0:5]))
pdf.savefig()

toplot['exp_B'] = exp_Base
toplot['exp_A'] = exp_Anes
toplot['ent_B'] = entr_Base
toplot['ent_A'] = entr_Anes
toplot['std_B'] = std_Base
toplot['std_A'] = std_Anes
toplot['ID'] = P_IDS

toplot_long = toplot.melt(id_vars=['outcome'],value_vars=['ent_B','ent_A'])
toplot_long.columns = ['outcome','phase','entr']

plt.figure()
sns.boxplot(x = 'outcome', y = 'entr',hue = 'phase', data = toplot_long)
sns.stripplot(x = 'outcome', y = 'entr',hue = 'phase', size=4, color=".3",data = toplot_long)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("Entropy")
pdf.savefig()

toplot_long = toplot.melt(id_vars=['outcome'],value_vars=['exp_B','exp_A'])
toplot_long.columns = ['outcome','phase','exp']

plt.figure()
sns.boxplot(x = 'outcome', y = 'exp',hue = 'phase', data = toplot_long)
sns.stripplot(x = 'outcome', y = 'exp',hue = 'phase', size=4, color=".3",data = toplot_long)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("Explained_variance 3 comp")
pdf.savefig()

toplot_long = toplot.melt(id_vars=['outcome'],value_vars=['std_B','std_A'])
toplot_long.columns = ['outcome','phase','std']

plt.figure()
sns.boxplot(x = 'outcome', y = 'std',hue = 'phase', data = toplot_long)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("std")
pdf.savefig()

difference_exp = toplot[['exp_A','exp_B']].max(axis=1)-toplot[['exp_A','exp_B']].min(axis=1)
difference_ent = toplot[['ent_A','ent_B']].max(axis=1)-toplot[['ent_A','ent_B']].min(axis=1)
difference = toplot[['std_A','std_B']].max(axis=1)-toplot[['std_A','std_B']].min(axis=1)

toplot['difference_exp'] = difference_exp
toplot['difference_ent'] = difference_ent
toplot['difference'] = difference

plt.figure()
sns.boxplot(x = 'outcome', y = 'difference', data = toplot)
sns.stripplot(x = 'outcome', y = 'difference', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("difference std")
pdf.savefig()

plt.figure()
sns.boxplot(x = 'outcome', y = 'difference_exp', data = toplot)
sns.stripplot(x = 'outcome', y = 'difference_exp', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("difference explained var")
pdf.savefig()

plt.figure()
sns.boxplot(x = 'outcome', y = 'difference_ent', data = toplot)
sns.stripplot(x = 'outcome', y = 'difference_ent', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("difference ent")
pdf.savefig()

# Make the plot
plt.figure()
parallel_coordinates(toplot[['ent_A','ent_B','outcome']], 'outcome', colormap=plt.get_cmap("Set2"))
pdf.savefig()

plt.figure()
parallel_coordinates(toplot[['std_A','std_B','outcome']], 'outcome', colormap=plt.get_cmap("Set2"))
pdf.savefig()


pdf.close()
