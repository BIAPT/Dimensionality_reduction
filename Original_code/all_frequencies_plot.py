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
            "PLOT_{}_{}_{}.pdf".format(frequency, MODE, STEP))

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

            data_on = np.mean(data_on, axis=0)
            data_off = np.mean(data_off, axis=0)

            # plot connectivity matrix
            figure = plt.figure(figsize=(19,10))
            plt.subplot(121)
            plt.imshow(data_off, cmap='jet')
            plt.title(p_id + '  off sedation')
            plt.clim(0,0.25)
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(data_on, cmap='jet')
            plt.title(p_id + '  on sedation')
            plt.clim(0,0.25)
            plt.colorbar()
            pdf.savefig(figure)
            plt.close()
            plt.show()

            avg_on, missing = average_matrices.extract_average_features(data_on, channels_on, p_id)
            avg_off, missing = average_matrices.extract_average_features(data_off, channels_off, p_id)

            visualize.plot_connectivity(avg_off)
            pdf.savefig()
            plt.close()

            visualize.plot_connectivity(avg_on)
            #plt.text(-20, 0.95, p_id + " off sedation " + MODE)
            pdf.savefig()
            plt.close()

        print('Mode finished {}  {}'.format(MODE, frequency))
        pdf.close()