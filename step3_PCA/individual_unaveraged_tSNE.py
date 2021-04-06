import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy.io import loadmat
from sklearn.manifold import TSNE
import seaborn as sns

mode = "dpli"
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
pdf = matplotlib.backends.backend_pdf.PdfPages("individual_unaveraged_tSNE_BA_{}.pdf".format(mode))

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
    data_Base = data_Base.transpose(2, 1, 0)
    data_Anes = data_Anes.transpose(2, 1, 0)
    # EARLIER IT WAS THIS: THIS IS AN ERROR:
    # data_Anes.reshape(electrodes,electrodes,data_Anes.shape[0])

    # transform to long data
    data_Base_long = np.transpose(data_Base[np.triu_indices(electrodes-1)])
    data_Anes_long = np.transpose(data_Anes[np.triu_indices(electrodes-1)])

    data = np.vstack((data_Base_long,data_Anes_long))
    label = np.ones(len(data))
    label[0:len(data_Base_long)] = 0
    phases = ['Base','Anes']

    data_2 = TSNE(n_components=2).fit_transform(data)

    fig = plt.figure(figsize=(10,10))
    scatter = plt.scatter(data_2[:, 0], data_2[:,1],marker='o', c=label)
    plt.scatter(data_2[:, 0], data_2[:,1],marker='o', c=label)
    plt.legend(handles=scatter.legend_elements()[0], labels=phases)
    plt.title(p_id)
    pdf.savefig()
    plt.close()

    data_3 = TSNE(n_components=3).fit_transform(data)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    scatter = ax.scatter(data_3[:, 0], data_3[:, 1],data_3[:, 2], marker='o', c=label)
    ax.scatter(data_3[:, 0], data_3[:, 1],data_3[:, 2], marker='o', c=label)
    ax.plot(data_3[:, 0], data_3[:, 1],data_3[:, 2],linewidth=0.6,color='k')
    ax.set_title(p_id)
    pdf.savefig()
    plt.close()

    data_1 = TSNE(n_components=1).fit_transform(data)

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

plt.figure()
sns.boxplot(x = 'outcome', y = 'Distance', data = toplot)
sns.stripplot(x = 'outcome', y = 'Distance', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['non-recovered','recoveded','healthy','NET_ICU'])
plt.title("Distance")
pdf.savefig()

pdf.close()


