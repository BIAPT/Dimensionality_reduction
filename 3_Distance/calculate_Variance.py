import scipy
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.decomposition import PCA
from scipy.spatial import distance
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

mode = 'dpli' # type of functional connectivity: can be dpli/ wpli
frequency = 'alpha' # frequency band: can be alpha/ theta/ delta
step = '10' # stepsize: can be '1'

IDS = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                        'S18', 'S19', 'S20', 'S22', 'S23',
                        'W03', 'W04', 'W08', 'W22', 'W28', 'W31', 'W34', 'W36',
                        'A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']

# define outcome:
# 0 = WSAS non-recovered
# 1 = WSAS recovered
# 2 = NET_ICU non-recovered
# 3 = NET_ICU recovered
# 4 = NET_ICU unknown
# 5 = Healthy

outcome = ['0', '1', '1', '0', '0', '0', '0','0', '1', '1', '0',
         '2', '3', '2', '3', '4', '2', '3', '3', '3',
         '5', '5', '5', '5', '5']

for frequency in frequencies:
    for MODE in MODES:
        for STEP in STEPS:

            INPUT_DIR = "../data/connectivity/{}/{}/step{}/".format(frequency, MODE, STEP)
            pdf = matplotlib.backends.backend_pdf.PdfPages(
                "Unaveraged_Variance_{}_{}_{}.pdf".format(frequency, MODE,STEP))

            PCA_variance = []
            PCA_variance_on = []
            PCA_variance_off = []
            variance_off = []
            variance_all = []
            variance_on = []

            for p_id in IDS:
                """
                1)    IMPORT DATA
                """

                if p_id.__contains__('MW') or p_id.__contains__('MG'):
                        OFF = 'sedoff'
                        ON = 'sedon1'
                if p_id.__contains__('WSAS'):
                        OFF = 'Base'
                        ON = 'Anes'
                if p_id.__contains__('MDFA'):
                        OFF = 'Base'
                        ON = 'indlast5'

                # define path for data ON and OFF
                data_on_path = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(MODE[0], frequency, STEP, p_id, ON)
                channels_on_path = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(MODE[0], frequency, STEP, p_id, ON)
                data_off_path = INPUT_DIR + "{}PLI_{}_step{}_{}_{}.mat".format(MODE[0], frequency, STEP, p_id, OFF)
                channels_off_path = INPUT_DIR + "{}PLI_{}_step{}_{}_{}_channels.mat".format(MODE[0], frequency, STEP, p_id, OFF)

                # load .mat and extract data
                data_on = loadmat(data_on_path)
                data_on = data_on["{}pli_tofill".format(MODE[0])]
                channel_on = scipy.io.loadmat(channels_on_path)['channels'][0][0]

                data_off = loadmat(data_off_path)
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
                common_channels = np.intersect1d(channels_on,channels_off)
                select_on = pd.Series(channels_on).isin(common_channels)
                select_off = pd.Series(channels_off).isin(common_channels)

                data_on = data_on[:, select_on, :]
                data_on = data_on[:, :, select_on]

                data_off = data_off[:, select_off, :]
                data_off = data_off[:, :, select_off]

                # extract the upper triangle and put the data into the 2D format
                nr_features = len(data_on[0][np.triu_indices(len(common_channels)-1)])
                nr_timesteps = min(len(data_off),len(data_on))

                data_on_2d = np.empty((nr_timesteps,nr_features))
                data_off_2d = np.empty((nr_timesteps,nr_features))

                for i in range(nr_timesteps):
                    data_on_2d[i] = data_on[i][np.triu_indices(len(common_channels)-1)]
                    data_off_2d[i] = data_off[i][np.triu_indices(len(common_channels)-1)]

                # combine the data over time
                data_combined = np.vstack((data_on_2d, data_off_2d))

                # define y-value (On/off) to combine data
                Y = np.zeros(len(data_combined))   # 0 is OFF
                Y[0:nr_timesteps] = 1              # 1 is ON

                pca = PCA().fit(data_combined)
                PCA_variance.append(sum(pca.explained_variance_))

                pca = PCA().fit(data_combined[Y==0])
                PCA_variance_off.append(sum(pca.explained_variance_))

                pca = PCA().fit(data_combined[Y==1])
                PCA_variance_on.append(sum(pca.explained_variance_))

                variance_all.append(np.var(data_combined))
                variance_on.append(np.var(data_combined[Y==1]))
                variance_off.append(np.var(data_combined[Y==0]))


            toplot = pd.DataFrame()
            toplot['ID'] = IDS
            toplot['outcome'] = outcome
            toplot['PCA Dim var all'] = PCA_variance
            toplot['PCA Dim var Base'] = PCA_variance_off
            toplot['PCA Dim var Anes'] = PCA_variance_on
            toplot['var all'] = variance_all
            toplot['var Base'] = variance_off
            toplot['var Anes'] = variance_on

            # 0 = WSAS non-recovered
            # 1 = WSAS recovered
            # 2 = NET_ICU non-recovered
            # 3 = NET_ICU recovered
            # 4 = NET_ICU unknown
            # 5 = Healthy


            for i in toplot.columns[2:]:
                plt.figure()
                sns.boxplot(x = 'outcome', y = i, data = toplot)
                sns.stripplot(x = 'outcome', y = i, size=4, color=".3",data = toplot)
                plt.xticks([0, 1, 2, 3, 4, 5], ['W_NR', 'W_R', 'N_NR',
                                                'N_R', 'N_U', 'HC'])

                plt.title( i )
                pdf.savefig()
                plt.close()

            pdf.close()



