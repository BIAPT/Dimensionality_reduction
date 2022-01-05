import scipy
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns

mode = 'AEC' # type of functional connectivity: can be dpli/ wpli / AEC
frequencies = ['theta','alpha','delta','beta'] # frequency band: can be alpha/ theta/ delta
conditions = ['Base', 'Anes']
step = '01' # stepsize: can be '01' or '10'

AllPart ={}

AllPart["Part"] = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
                   'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22']

AllPart["Part_nonr"] = ['WSAS05', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13', 'WSAS18',
                        'WSAS22']

AllPart["Part_reco"] = ['WSAS02', 'WSAS09','WSAS19', 'WSAS20']

IDS = AllPart["Part"]
outcome = []
group = []


for i,p in enumerate(AllPart["Part"]):
    if AllPart["Part_nonr"].__contains__(p):
        outcome.append(1)
        group.append("nonr")
    if AllPart["Part_reco"].__contains__(p):
        outcome.append(0)
        group.append("reco")

pdf = matplotlib.backends.backend_pdf.PdfPages("VAll_frequencies_{}_{}.pdf".format(mode, step))

Base_mean = pd.DataFrame()
Base_std_time = pd.DataFrame()
Base_std_space = pd.DataFrame()

Anes_mean = pd.DataFrame()
Anes_std_time = pd.DataFrame()
Anes_std_space = pd.DataFrame()

for cond in conditions:
    for frequency in frequencies:
        INPUT_DIR = "data/FC/{}/{}/step{}/".format(frequency, mode, step)
        # define features to extract from raw data
        Mean = []
        # variance over time, space and all on and off
        Var_time = []
        Var_space = []

        for p_id in IDS:

            """
            1)    IMPORT DATA
            """
            if mode == 'AEC':
                # define path for data ON and OFF
                data_path = INPUT_DIR + "AEC_{}_step{}_{}_{}.mat".format(frequency, step, p_id,cond)
                channels_path = INPUT_DIR + "AEC_{}_step{}_{}_{}_channels.mat".format(frequency, step, p_id,cond)
            else:
                # define path for data ON and OFF
                data_path = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(mode[0], frequency, step, p_id,cond)
                channels_path = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(mode[0], frequency, step, p_id,cond)

            # load .mat and extract data
            data = loadmat(data_path)
            if mode == "AEC":
                data = data["aec_tofill"]
            else:
                data = data["{}pli_tofill".format(mode[0])]
            channel = scipy.io.loadmat(channels_path)['channels'][0][0]
            print('Load data comlpete {} {}'.format(p_id, cond))

            # extract channels from the weird format
            channels = []
            for a in range(0, len(channel)):
                channels.append(channel[a][0])
            channels = np.array(channels)

            # reduce channels to avoid nonbrain channels
            nonbrain = ["E127", "E126", "E17", "E21", "E14", "E25", "E8", "E128", "E125", "E43", "E120", "E48", "E119", "E49",
                        "E113", "E81", "E73", "E88", "E68", "E94", "E63", "E99", "E56", "E107"]
            common_channels = np.intersect1d(channels, nonbrain)
            # only channels which are not in nobrain
            select = np.invert(pd.Series(channels).isin(common_channels))
            channels = channels[select]

            data = data[:, select, :]
            data = data[:, :, select]

            # extract the upper triangle and put the data into the 2D format
            nr_features = len(data[0][np.triu_indices(len(channels),k=1)])
            nr_timesteps = len(data)

            data_2d = np.empty((nr_timesteps, nr_features))

            for i in range(nr_timesteps):
                data_2d[i] = data[i][np.triu_indices(len(channels),k=1)]

            """
            Calculate Mean and Variance
            """
            Mean.append(np.mean(data_2d))

            variance = np.std(np.mean(data_2d,axis=1))
            Var_time.append(variance)

            variance = np.std(np.mean(data_2d,axis=0))
            Var_space.append(variance)

        if cond == 'Base':
            Base_mean[frequency]= Mean
            Base_std_time[frequency]= Var_time
            Base_std_space[frequency]= Var_space

        if cond == 'Anes':
            Anes_mean[frequency]= Mean
            Anes_std_time[frequency]= Var_time
            Anes_std_space[frequency]= Var_space


def spiderplot(toplot,Name,frequencies):
    toplot['outcome'] = outcome
    toplot['group'] = group

    Nonreco_all = (toplot[toplot.outcome == 1])[frequencies]
    Reco_all = (toplot[toplot.outcome == 0])[frequencies]

    Reco = np.mean(Reco_all)
    Nonreco = np.mean(Nonreco_all)

    Nonreco = [*Nonreco, Nonreco[0]]
    Reco = [*Reco, Reco[0]]
    frequencies = [*frequencies, frequencies[0]]

    Nonreco_all = Nonreco_all[frequencies]
    Reco_all = Reco_all[frequencies]

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=5)

    fig = plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    plt.plot(label_loc, Reco, label='Recovered')
    plt.fill(label_loc, Reco, 'skyblue', alpha=0.4)
    for i in range(0,len(Reco_all)):
        plt.scatter(label_loc,Reco_all.iloc[i],c='blue')
    plt.plot(label_loc, Nonreco, label='Non-Recovered')
    plt.fill(label_loc, Nonreco, 'orange', alpha=0.3)
    for i in range(0,len(Nonreco_all)):
        plt.scatter(label_loc,Nonreco_all.iloc[i],c='orange')
    plt.title(Name, size=20, y=1.05)
    plt.thetagrids(np.degrees(label_loc), labels=frequencies)
    plt.legend()
    pdf.savefig(fig)
    plt.close()


spiderplot(Base_mean, "{}_Base_mean".format(mode), frequencies)
spiderplot(Base_std_space , "{}_Base_std_space".format(mode), frequencies)
spiderplot(Base_std_time , "{}_Base_std_time".format(mode), frequencies)
spiderplot(Anes_mean, "{}_Anes_mean".format(mode), frequencies)
spiderplot(Anes_std_space, "{}_Anes_sd_space".format(mode), frequencies)
spiderplot(Anes_std_time, "{}_Anes_sd_time".format(mode), frequencies)

pdf.close()