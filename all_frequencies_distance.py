import scipy
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd

from utils import visualize

sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA
from scipy.spatial import distance
import seaborn as sns
from utils import average_matrices as average_matrices

MODES = ['AEC','wPLI'] # type of functional connectivity: can be dpli/ wpli / AEC
frequencies = ['theta','alpha','delta','beta'] # frequency band: can be alpha/ theta/ delta
conditions = ['Base', 'Anes']
STEP = '01' # stepsize: can be '01' or '10'
dist_type = "euclidean"  # "euclidean"

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
        outcome.append('1')
        group.append("nonr")
    if AllPart["Part_reco"].__contains__(p):
        outcome.append('0')
        group.append("reco")

for frequency in frequencies:
    for MODE in MODES:

        INPUT_DIR = "data/FC/{}/{}/step{}/".format(frequency, MODE, STEP)
        pdf = matplotlib.backends.backend_pdf.PdfPages(
            "SNACC_Unaveraged_Distance_{}_{}_{}.pdf".format(frequency, MODE, STEP))

        # mean of on and off
        Mean_on = []
        Mean_off = []

        # variance over time, space and all on and off
        Var_on_time = []
        Var_on_space = []
        Var_on = []
        Var_off_time = []
        Var_off_space = []
        Var_off = []

        # overall distance
        Dist_overall = []

        # between distance
        Dist_bet_raw = []
        Dist_bet_norm = []
        Dist_bet_norm_base = []

        # within distance
        Dist_w_on_raw = []
        Dist_w_off_raw = []
        Dist_w_Base_n_Anes = []
        Dist_w_on = []
        Dist_w_off = []

        PC1_weights = []

        for p_id in IDS:
            """
            1)    IMPORT DATA
            """
            if p_id.__contains__('WSAS'):
                OFF = 'Base'
                ON = 'Anes'
            if p_id.__contains__('MDFA'):
                OFF = 'Base'
                ON = 'indlast5'

            # define path for data ON and OFF
            data_on_path = INPUT_DIR + "{}_{}_step{}_{}_{}.mat".format(MODE, frequency, STEP, p_id, ON)
            channels_on_path = INPUT_DIR + "{}_{}_step{}_{}_{}_channels.mat".format(MODE, frequency, STEP,
                                                                                       p_id, ON)
            data_off_path = INPUT_DIR + "{}_{}_step{}_{}_{}.mat".format(MODE, frequency, STEP, p_id, OFF)
            channels_off_path = INPUT_DIR + "{}_{}_step{}_{}_{}_channels.mat".format(MODE, frequency, STEP,
                                                                                        p_id, OFF)

            # load .mat and extract data
            data_on = loadmat(data_on_path)
            if MODE == 'AEC':
                data_on = data_on["aec_tofill"]
            else:
                data_on = data_on["{}pli_tofill".format(MODE[0])]

            channel_on = scipy.io.loadmat(channels_on_path)['channels'][0][0]

            data_off = loadmat(data_off_path)
            if MODE == 'AEC':
                data_off = data_off["aec_tofill"]
            else:
                data_off = data_off["{}pli_tofill".format(MODE[0])]

            channel_off = scipy.io.loadmat(channels_off_path)['channels'][0][0]

            print('Load data comlpete {}'.format(p_id))

            # extract channels from the weird format
            channels_on = []
            channels_off = []
            for a in range(0, len(channel_on)):
                channels_on.append(channel_on[a][0])
            for a in range(0, len(channel_off)):
                channels_off.append(channel_off[a][0])
            channels_on = np.array(channels_on)
            channels_off = np.array(channels_off)

            # reduce the baseline and anesthesia data to the same channel subset
            common_channels = np.intersect1d(channels_on, channels_off)
            select_on = pd.Series(channels_on).isin(common_channels)
            select_off = pd.Series(channels_off).isin(common_channels)

            data_on = data_on[:, select_on, :]
            data_on = data_on[:, :, select_on]

            data_off = data_off[:, select_off, :]
            data_off = data_off[:, :, select_off]

            # extract the upper triangle and put the data into the 2D format
            nr_features = len(data_on[0][np.triu_indices(len(common_channels),k=1)])
            nr_timesteps = min(len(data_off), len(data_on))

            data_on_2d = np.empty((nr_timesteps, nr_features))
            data_off_2d = np.empty((nr_timesteps, nr_features))

            for i in range(nr_timesteps):
                data_on_2d[i] = data_on[i][np.triu_indices(len(common_channels),k=1)]
                data_off_2d[i] = data_off[i][np.triu_indices(len(common_channels),k=1)]

            # combine the data over time
            data_combined = np.vstack((data_on_2d, data_off_2d))
            # define y-value (On/off) to combine data
            Y = np.zeros(len(data_combined))  # 0 is OFF
            Y[0:nr_timesteps] = 1  # 1 is ON

            """
            Calculate Mean and Variance
            """
            Mean_on.append(np.mean(data_on_2d))
            Var_on_time.append(np.mean(np.var(data_on_2d,axis=0)))
            Var_on_space.append(np.mean(np.var(data_on_2d,axis=1)))
            Var_on.append(np.var(data_on_2d))

            Mean_off.append(np.mean(data_off_2d))
            Var_off_time.append(np.mean(np.var(data_off_2d,axis=0)))
            Var_off_space.append(np.mean(np.var(data_off_2d,axis=1)))
            Var_off.append(np.var(data_off_2d))


            """
            2)    RUN PCA
            """
            pca_2 = PCA(n_components=2).fit(data_combined)
            data_combined_2 = pca_2.transform(data_combined)

            plt.figure()
            n = np.where(Y == 0)
            plt.scatter(data_combined_2[n, 0], data_combined_2[n, 1], c='red', label='OFF')
            n = np.where(Y == 1)
            plt.scatter(data_combined_2[n, 0], data_combined_2[n, 1], c='blue', label='ON')
            #plt.legend()
            #plt.title(p_id)
            plt.xlabel('PC 1')
            plt.ylabel('PC 2')
            plt.savefig("PCA_{}.jpeg".format(p_id))
            pdf.savefig()
            plt.close()

            """
            Run PCA 1D
            --> Component Features
            """
            pca_1 = PCA(n_components=1).fit(data_combined)
            features = pca_1.components_[0]
            feature_matrix = np.empty((len(common_channels),len(common_channels)))
            feature_matrix[np.triu_indices(len(common_channels),k=1)] = features

            avg_features, missing = average_matrices.extract_average_features(feature_matrix , common_channels, p_id)
            PC1_weights.append(avg_features.values[0])

            norm_features= (avg_features.values - np.min(avg_features.values)) / \
                          (np.max(avg_features.values) - np.min(avg_features.values))

            avg_features[:] = norm_features

            visualize.plot_features(avg_features)
            #plt.text(-20, 0.95, p_id + "  " + MODE)
            plt.savefig("Feature_{}.jpeg".format(p_id))
            pdf.savefig()
            plt.close()

            """
            Calculate Distances
            """
            # distance over all points
            overall_dist = distance.cdist(data_combined, data_combined, dist_type)

            # plot overall distance
            plt.figure()
            plt.imshow(overall_dist)
            plt.colorbar()
            plt.title("OVERALL Distance " + p_id)
            #plt.clim(0, 20)
            plt.savefig("Distance_{}.jpeg".format(p_id))
            pdf.savefig()
            plt.close()

            # this matrix is symmetric. Therefore/2
            overall_dist = np.sum(overall_dist) / 2
            Dist_overall.append(overall_dist)

            # distance between ON and OFF
            # in percent dependent on overall dist
            bet_dist = distance.cdist(data_combined[Y == 0], data_combined[Y == 1], dist_type)
            Dist_bet_raw.append(np.sum(bet_dist))
            Dist_bet_norm.append((np.sum(bet_dist) / overall_dist) * 100)

            # distance within ON and OFF
            dist_within_on = distance.cdist(data_combined[Y == 1], data_combined[Y == 1], dist_type)
            dist_within_off = distance.cdist(data_combined[Y == 0], data_combined[Y == 0], dist_type)
            # due to symmetry /2
            dist_within_on = np.sum(dist_within_on) / 2
            dist_within_off = np.sum(dist_within_off) / 2

            Dist_w_Base_n_Anes.append((dist_within_off / dist_within_on) * 100)
            Dist_bet_norm_base.append((np.sum(bet_dist) / dist_within_off)*100)

            # save unnormalized distance
            Dist_w_on_raw.append(dist_within_on)
            Dist_w_off_raw.append(dist_within_off)

            # normalize by ovrall distance
            dist_within_on = (dist_within_on / overall_dist) * 100
            dist_within_off = (dist_within_off / overall_dist) * 100

            Dist_w_on.append(dist_within_on)
            Dist_w_off.append(dist_within_off)

        toplot = pd.DataFrame()
        toplot['ID'] = IDS
        toplot['outcome'] = outcome

        #mean
        toplot['Mean Base'] = Mean_off
        toplot['Mean Anes'] = Mean_on

        #variance
        toplot['Variance time Base'] = Var_off_time
        toplot['Variance time Anes'] = Var_on_time
        toplot['Variance space Base'] = Var_off_space
        toplot['Variance space Anes'] = Var_on_space
        toplot['Variance all Base'] = Var_off
        toplot['Variance all Anes'] = Var_on

        toplot['Overall Distance'] = Dist_overall
        toplot['Unnorm Distance between Base-Anes'] = Dist_bet_raw
        toplot['NORM Distance between Base-Anes'] = Dist_bet_norm
        toplot['Base NORM Distance between Base-Anes'] = Dist_bet_norm_base

        toplot['Unnorm Distance within Base'] = Dist_w_off_raw
        toplot['NORM Distance within Base'] = Dist_w_off
        toplot['Unnorm Distance within Anes'] = Dist_w_on_raw
        toplot['NORM Distance within Anes'] = Dist_w_on
        toplot['Distance within Base/Anes'] = Dist_w_Base_n_Anes

        # 0 = WSAS non-recovered
        # 1 = WSAS recovered
        # 2 = Healthy

        for i in toplot.columns[2:]:
            plt.figure()
            sns.boxplot(x='outcome', y=i, data=toplot)
            sns.stripplot(x='outcome', y=i, size=4, color=".3", data=toplot)
            plt.xticks([0, 1 ], ['Reco', 'NonReco'])
            plt.title(i)
            pdf.savefig()
            plt.close()

        # plot averaged weights
        PC1_weights = pd.DataFrame(PC1_weights)
        nonr_weight = PC1_weights[np.array(outcome) == '1']
        reco_weight = PC1_weights[np.array(outcome) == '0']

        # plot average weights normalized
        areas = avg_features.columns

        #RECOVERED
        mean_reco = np.array(np.mean(reco_weight))
        mean_reco_norm = (mean_reco - np.min(mean_reco)) / (np.max(mean_reco) - np.min(mean_reco))
        features_reco = pd.DataFrame(mean_reco_norm.reshape(-1, len(areas)), columns=areas)
        visualize.plot_features(features_reco)
        plt.savefig("Feature_Reco.jpeg")
        pdf.savefig()
        plt.close()

        # NON_Recovered
        mean_nonr = np.array(np.mean(nonr_weight))
        mean_nonr_norm = (mean_nonr - np.min(mean_nonr)) / (np.max(mean_nonr) - np.min(mean_nonr))
        features_nonr = pd.DataFrame(mean_nonr_norm.reshape(-1, len(areas)), columns=areas)
        visualize.plot_features(features_nonr)
        plt.savefig("Feature_Nonreco.jpeg")
        pdf.savefig()
        plt.close()

        toplot.to_csv("SNACC_{}_{}_{}.csv".format(frequency, MODE, STEP), index=False, sep=';')

        pdf.close()