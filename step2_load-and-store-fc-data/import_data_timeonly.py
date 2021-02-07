"""
written by CHARLOTTE MASCHKE for: DOC Dimensionality reduction 2021
This code contains a list of participants and a list of electrodes which belong to one areas of interest
It will go through the individual wPLI and dPLI files and select the features (functional connectivity for the regions
of interest).

It will output a pickle and csv with the features for the ML pipeline
"""

import scipy
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd
sys.path.append('../')
from utils import extract_features


# Loop over these parameters
#FREQUENCY = ["alpha", "theta", "delta"]
#STEP = ["10", "01"]
frequency = "alpha"
STEP = ["01","10"]
MODE = ["wpli", "dpli"]
CONDITION = ["Base", "Anes"]

P_IDS = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
         'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22',
         'MCD0004', 'MCD0007', 'MCD0008', 'MCD0009', 'MCD0012', 'MCD0013','MCD0014', 'MCD0018', 'MCD0021',
         '002MG', '003MG', '004MG', '004MW']

ROI = ['LF_LC', 'LF_LP', 'LF_LO', 'LF_LT',
       'LT_LO', 'LT_LC', 'LT_LP',
       'LP_LO', 'LP_LC',
       'LC_LO',
       'LF_LF', 'LC_LC', 'LP_LP', 'LT_LT', 'LO_LO',

       'RF_RC', 'RF_RP', 'RF_RO', 'RF_RT',
       'RT_RO', 'RT_RC', 'RT_RP',
       'RP_RO', 'RP_RC',
       'RC_RO',
       'RF_RF', 'RC_RC', 'RP_RP', 'RT_RT', 'RO_RO',

       'LF_RC', 'LF_RP', 'LF_RO', 'LF_RT',
       'LT_RO', 'LT_RC', 'LT_RP',
       'LP_RO', 'LP_RC',
       'LC_RO',

       'RF_LC', 'RF_LP', 'RF_LO', 'RF_LT',
       'RT_LO', 'RT_LC', 'RT_LP',
       'RP_LO', 'RP_LC',
       'RC_LO',

       'LF_RF', 'LC_RC', 'LP_RP', 'LT_RT', 'LO_RO']

for step in STEP:
    for mode in MODE:
        # for Beluga
        #OUTPUT_DIR = "/home/lotte/projects/def-sblain/lotte/Dim_DOC/results/features/"
        #INPUT_DIR = "/home/lotte/projects/def-sblain/lotte/Dim_DOC/results/{}/{}/step{}/".format(frequency, mode, step)
        OUTPUT_DIR = "../data/features/"
        INPUT_DIR = "../data/connectivity/{}/{}/step{}/".format(frequency, mode,step)
        # empty dataframe for all participants
        df_wpli_final = pd.DataFrame()

        for p_id in P_IDS:
            for cond in CONDITION:
                # change the condition name for other groups
                if p_id.__contains__('MW') or p_id.__contains__('MG'):
                    if cond == 'Base':
                        cond = 'Sedoff'
                    if cond == 'Anes':
                        cond = 'Sedon1'
                if p_id.__contains__('MCD'):
                    if cond == 'Base':
                        cond = 'eyesclosed1'
                    if cond == 'Anes':
                        cond = 'emergencefirst5'

                part_in = INPUT_DIR +"{}PLI_{}_step{}_{}_{}.mat".format(mode[0], frequency, step, p_id, cond)
                part_channels = INPUT_DIR +"{}PLI_{}_step{}_{}_{}_channels.mat".format(mode[0], frequency, step, p_id, cond)

                data = loadmat(part_in)
                data = data["{}pli_tofill".format(mode[0])]
                channel = scipy.io.loadmat(part_channels)['channels'][0][0]
                print('Load data comlpete {}'.format(part_in))

                channels = []
                for a in range(0,len(channel)):
                    channels.append(channel[a][0])
                channels = np.array(channels)

                # change the channel notation from 'Fp2 to E9'
                if p_id != 'WSAS02':
                    channels[np.where(channels == 'Fp2')] = 'E9'
                    channels[np.where(channels == 'Fz')] = 'E11'
                    channels[np.where(channels == 'F3')] = 'E24'
                    channels[np.where(channels == 'F7')] = 'E33'
                    channels[np.where(channels == 'Fp1')] = 'E22'
                    channels[np.where(channels == 'C3')] = 'E36'
                    channels[np.where(channels == 'T7')] = 'E25'
                    channels[np.where(channels == 'P3')] = 'E52'
                    channels[np.where(channels == 'LM')] = 'E57'
                    channels[np.where(channels == 'P7')] = 'E58'
                    channels[np.where(channels == 'Pz')] = 'E62'
                    channels[np.where(channels == 'O1')] = 'E70'
                    channels[np.where(channels == 'Oz')] = 'E75'
                    channels[np.where(channels == 'O2')] = 'E83'
                    channels[np.where(channels == 'P4')] = 'E92'
                    channels[np.where(channels == 'P8')] = 'E96'
                    channels[np.where(channels == 'RM')] = 'E100'
                    channels[np.where(channels == 'C4')] = 'E104'
                    channels[np.where(channels == 'T8')] = 'E108'
                    channels[np.where(channels == 'F8')] = 'E122'
                    channels[np.where(channels == 'F4')] = 'E124'

                # create empty dataframe to fill
                # name wpli but can also contain dpli, depending on mode
                ID = p_id[3:6]

                names = ROI.copy()
                names.insert(0, 'Name')
                names.insert(1, 'ID')
                names.insert(2, 'Phase')
                names.insert(3, 'Time')

                missingel = []
                time_steps=data.shape[0]

                df_wpli = pd.DataFrame(np.zeros((time_steps,len(names))))

                name = "{}_{}".format(p_id,cond)
                df_wpli.iloc[:,0] = name
                df_wpli.iloc[:,1] = p_id
                df_wpli.iloc[:,2] = cond

                # initialize a dict of regions and referring electrodes
                regions = {}

                regions["LF"] = ['E15', 'E32', 'E22', 'E16', 'E18', 'E23', 'E26', 'E11', 'E19', 'E24', 'E27', 'E33',
                                 'E12', 'E20', 'E28', 'E34']
                regions["LC"] = ['E6', 'E13', 'E29', 'E35', 'E7', 'E30', 'E36', 'E41', 'Cz', 'E31', 'E37', 'E42',
                                 'E55', 'E54', 'E47', 'E53']
                regions["LP"] = ['E52', 'E51', 'E61', 'E62', 'E60', 'E67', 'E59', 'E72', 'E58', 'E71', 'E66']
                regions["LO"] = ['E75', 'E70', 'E65', 'E64', 'E74', 'E69']
                regions["LT"] = ['E38', 'E44', 'E39', 'E40', 'E46', 'E45', 'E50', 'E57']

                regions["RF"] = ['E15', 'E1', 'E9', 'E16', 'E10', 'E3', 'E2', 'E11', 'E4', 'E124', 'E123', 'E122',
                                 'E5', 'E118', 'E117', 'E116']
                regions["RC"] = ['E6', 'E112', 'E111', 'E110', 'E106', 'E105', 'E104', 'E103', 'Cz', 'E80', 'E87',
                                 'E93', 'E55', 'E79', 'E98', 'E86']
                regions["RP"] = ['E92', 'E97', 'E78', 'E62', 'E85', 'E77', 'E91', 'E72', 'E96', 'E76', 'E84']
                regions["RO"] = ['E75', 'E83', 'E90', 'E95', 'E82', 'E89']
                regions["RT"] = ['E121', 'E114', 'E115', 'E109', 'E102', 'E108', 'E101', 'E100']


                if p_id == 'WSAS02':

                    regions = {}
                    regions["LF"] = ['Fp1', 'Fpz', 'AF3', 'AF7', 'Afz', 'Fz', 'F1', 'F3', 'F5', 'F7']
                    regions["LC"] = ['Cz', 'C1', 'C3', 'C5', 'FCz', 'FC1', 'FC3', 'FC5']
                    regions["LP"] = ['Pz', 'P1','P3', 'P5', 'P7', 'CP1', 'CP3', 'CP5', 'CPz']
                    regions["LO"] = ['POz', 'PO3', 'PO7', 'Oz', 'O1']
                    regions["LT"] = ['FT7', 'T7', 'TP7']

                    regions['RF'] = ['Fp2', 'Fpz', 'AF4', 'AF8', 'Afz', 'Fz', 'F2', 'F4', 'F6', 'F8']
                    regions['RC'] = ['Cz', 'C2', 'C4', 'C6', 'FCz', 'FC2', 'FC4', 'FC6']
                    regions['RP'] = ['Pz', 'P2', 'P4', 'P6', 'P8', 'CP2', 'CP4', 'CP6', 'CPz']
                    regions['RO'] = ['POz', 'PO4', 'PO8', 'Oz', 'O2']
                    regions['RT'] = ['FT8', 'T8', 'TP8']

                for t in range(0, time_steps):
                    df_wpli.iloc[t, 3] = t
                    i = 4   # Position in DataFrame: 0-3 are 'Name','ID','Phase','Time'

                    for r in ROI:
                        r1=r[0:2]
                        r2=r[3:5]

                        conn, missing = extract_features.extract_single_features(X_step=data[t].copy(), channels=channels,
                                                                      selection_1=regions[r1], selection_2=regions[r2],
                                                                      time=t)
                        missingel.extend(missing)
                        df_wpli.iloc[t, i] = conn

                        i += 1
                df_wpli.columns = names
                df_wpli_final = pd.concat([df_wpli_final,df_wpli])
                print("Participant" + name + "   finished")
                print("missing electrodes: " + str(list(set(missingel))))

                df_wpli_final.columns = names

        df_wpli_final.to_pickle(OUTPUT_DIR + "23_Part_{}_10_{}_{}.pickle".format(mode, step, frequency), protocol=4)
        df_wpli_final.to_csv(OUTPUT_DIR + "23_Part_{}_10_{}_{}.csv".format(mode, step, frequency))

