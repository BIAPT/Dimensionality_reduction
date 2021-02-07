import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA
from scipy.io import loadmat

mode = "wpli"
frequency = "alpha"
step = "01"
CONDITION = ["Base", "Anes"]

P_IDS = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
         'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22',
         'MCD0004', 'MCD0007', 'MCD0008', 'MCD0009', 'MCD0012', 'MCD0013','MCD0014', 'MCD0018', 'MCD0021',
         '002MG', '003MG', '004MG', '004MW']

INPUT_DIR = "data/connectivity/{}/{}/step{}/".format(frequency, mode,step)

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

    data_Base = data_Base.reshape(electrodes,electrodes,data_Base.shape[0])
    data_Anes = data_Anes.reshape(electrodes,electrodes,data_Anes.shape[0])

    # transform to long data
    data_Base_long = np.transpose(data_Base[np.triu_indices(electrodes-1)])
    data_Anes_long = np.transpose(data_Anes[np.triu_indices(electrodes-1)])

    data = np.vstack((data_Base_long,data_Anes_long))
    label = np.ones(len(data))
    label[0:len(data_Base_long)] = 0
    phases = ['Base','Anes']

    pca = PCA(n_components=2)
    data_2 = pca.fit_transform(data)

    fig = plt.figure(figsize=(10,10))
    scatter = plt.scatter(data_2[:, 0], data_2[:,1],marker='o', c=label)
    plt.scatter(data_2[:, 0], data_2[:,1],marker='o', c=label)
    plt.legend(handles=scatter.legend_elements()[0], labels=phases)
    plt.title(p_id)
    pdf.savefig()
    plt.close()

    pca = PCA(n_components=3)
    data_3 = pca.fit_transform(data)

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    scatter = ax.scatter(data_3[:, 0], data_3[:, 1],data_3[:, 2], marker='o', c=label)
    ax.scatter(data_3[:, 0], data_3[:, 1],data_3[:, 2], marker='o', c=label)
    ax.plot(data_3[:, 0], data_3[:, 1],data_3[:, 2],linewidth=0.6,color='k')
    ax.set_title(p_id)
    pdf.savefig()
    plt.close()


pdf.close()


