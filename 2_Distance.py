import scipy
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd
import argparse
#from utils import visualize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance
import seaborn as sns
from utils import average_matrices as average_matrices
from matplotlib.backends.backend_pdf import PdfPages
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculates Disctance matrix from wPLI functional connectivity.')
    parser.add_argument('-input_dir', type=str, action='store',
                        help='folder name containing the data in epoched .fif data in BIDS format')
    parser.add_argument('-output_dir', type=str, action='store',
                        help='folder name where to save the functional connectivity')
    parser.add_argument('-participants', type=str, action='store',
                        help='path to txt with information about participants')
    parser.add_argument('-frequencyband', type=str,
                        help='lower and upper filer frequency')
    args = parser.parse_args()


    """
           1)    PREPARE IN-AND OUTPUT
    """
    # make ouput directory
    output_dir = os.path.join(args.output_dir, 'wPLI_Distance')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare output pdf
    pdf = PdfPages(f"{output_dir}/wPLI_Distance_{args.frequencyband}.pdf")

    # load patient IDS
    info = pd.read_csv(args.participants, sep='\t')
    P_IDS = info['Patient']
    outcome = info['Outcome_Prog']

    dist_type = "euclidean"  # "euclidean"

    #define empty results structure
    # mean of on and off
    Mean_Anes = []
    Mean_Base = []

    # variance over time, space and all on and off
    Var_Anes_time = []
    Var_Anes_space = []
    Var_Anes = []
    Var_Base_time = []
    Var_Base_space = []
    Var_Base = []

    # overall distance
    Dist_overall = []

    # between distance
    Dist_bet_raw = []
    Dist_bet_norm_all = []
    Dist_bet_norm_base = []
    Dist_bet_norm_base_anes = []

    # within distance
    Dist_w_Anes_raw = []
    Dist_w_Base_raw = []
    Dist_w_Base_n_Anes = []
    Dist_w_Anes_norm = []
    Dist_w_Base_norm = []

    PC1_weights = []


    for p_id in P_IDS:
        """
        1)    IMPORT DATA
        """
        wpli_Base = np.load(f"{args.input_dir}/wPLI_{args.frequencyband}/WPLI_{args.frequencyband}_{p_id}_Base.npy")
        wpli_Anes = np.load(f"{args.input_dir}/wPLI_{args.frequencyband}/WPLI_{args.frequencyband}_{p_id}_Anes.npy")

        print(p_id)
        print(wpli_Base.shape)
        print(wpli_Anes.shape)

        nr_channesl = wpli_Base.shape[1]
        nr_timesteps = min(wpli_Base.shape[0], wpli_Anes.shape[0])
        # extract the upper triangle and put the data into the 2D format
        nr_features = len(wpli_Base[0][np.triu_indices(nr_channesl,k=1)])

        # initialize empty dataframe for 2d Data
        data_Base_2d = np.empty((nr_timesteps, nr_features))
        data_Anes_2d = np.empty((nr_timesteps, nr_features))

        for i in range(nr_timesteps):
            data_Base_2d[i] = wpli_Base[i][np.triu_indices(nr_channesl,k=1)]
            data_Anes_2d[i] = wpli_Anes[i][np.triu_indices(nr_channesl,k=1)]

        # combine the data over time
        data_2d = np.vstack((data_Base_2d, data_Anes_2d))
        # define y-value (On/off) to combine data
        Y = np.ones(len(data_2d))  # 1 is Anes
        Y[0:nr_timesteps] = 0  # 0 is Base

        """
        Calculate Mean and Variance
        """
        Mean_Anes.append(np.mean(data_Anes_2d))
        Var_Anes_time.append(np.mean(np.var(data_Anes_2d,axis=0)))
        Var_Anes_space.append(np.mean(np.var(data_Anes_2d,axis=1)))
        Var_Anes.append(np.var(data_Anes_2d))

        Mean_Base.append(np.mean(data_Base_2d))
        Var_Base_time.append(np.mean(np.var(data_Base_2d,axis=0)))
        Var_Base_space.append(np.mean(np.var(data_Base_2d,axis=1)))
        Var_Base.append(np.var(data_Base_2d))


        """
        2)    RUN PCA
        """
        pca_2 = PCA(n_components=2).fit(data_2d)
        data_pca2 = pca_2.transform(data_2d)

        plt.figure()
        n = np.where(Y == 0)
        a = np.linspace(0.1, 1, len(n[0]))
        plt.scatter(data_pca2[n, 0], data_pca2[n, 1], alpha=a, c='red', label='Base')
        n = np.where(Y == 1)
        a = np.linspace(0.1, 1, len(n[0]))
        plt.scatter(data_pca2[n, 0], data_pca2[n, 1], alpha=a, c='blue', label='Anes')
        #plt.legend()
        plt.title(p_id)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        #plt.savefig("PCA_{}.jpeg".format(p_id))
        pdf.savefig()
        plt.close()

        """
        Run PCA 1D
        --> Component Features
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
        #plt.savefig("Feature_{}.jpeg".format(p_id))
        pdf.savefig()
        plt.close()
        """

        """
        Calculate Distances
        """
        # distance over all points
        overall_dist = distance.cdist(data_2d, data_2d, dist_type)

        # plot overall distance
        plt.figure()
        plt.imshow(overall_dist)
        plt.colorbar()
        plt.title("OVERALL Distance " + p_id)
        #plt.clim(0, 20)
        #plt.savefig("Distance_{}.jpeg".format(p_id))
        pdf.savefig()
        plt.close()

        # this matrix is symmetric. Therefore/2
        overall_dist = np.sum(overall_dist) / 2
        Dist_overall.append(overall_dist)

        # distance between ON and OFF
        # in percent dependent on overall dist
        bet_dist = distance.cdist(data_2d[Y == 0], data_2d[Y == 1], dist_type)
        Dist_bet_raw.append(np.sum(bet_dist))
        Dist_bet_norm_all.append((np.sum(bet_dist) / overall_dist) * 100)

        # distance within ON and OFF
        dist_within_Anes = distance.cdist(data_2d[Y == 1], data_2d[Y == 1], dist_type)
        dist_within_Base = distance.cdist(data_2d[Y == 0], data_2d[Y == 0], dist_type)
        # due to symmetry /2
        dist_within_Anes = np.sum(dist_within_Anes) / 2
        dist_within_Base = np.sum(dist_within_Base) / 2

        Dist_bet_norm_base.append((np.sum(bet_dist) / dist_within_Base)*100)
        Dist_bet_norm_base_anes.append((np.sum(bet_dist) / (dist_within_Base+dist_within_Anes))*100)

        # save unnormalized distance
        Dist_w_Anes_raw.append(dist_within_Anes)
        Dist_w_Base_raw.append(dist_within_Base)

        # normalize by ovrall distance
        dist_within_Anes = (dist_within_Anes / overall_dist) * 100
        dist_within_Base = (dist_within_Base / overall_dist) * 100

        Dist_w_Anes_norm.append(dist_within_Anes)
        Dist_w_Base_norm.append(dist_within_Base)

    toplot = pd.DataFrame()
    toplot['ID'] = P_IDS
    toplot['outcome'] = outcome

    #mean
    toplot['Mean Base'] = Mean_Base
    toplot['Mean Anes'] = Mean_Anes

    #variance
    toplot['Variance time Base'] = Var_Base_time
    toplot['Variance time Anes'] = Var_Anes_time
    toplot['Variance space Base'] = Var_Base_space
    toplot['Variance space Anes'] = Var_Anes_space
    toplot['Variance all Base'] = Var_Base
    toplot['Variance all Anes'] = Var_Anes

    toplot['Overall Distance'] = Dist_overall
    toplot['B_Dist_Base_Anes'] = Dist_bet_raw
    toplot['Norm_B_Dist_Base_Anes'] = Dist_bet_norm_all
    toplot['Basenorm_B_Dist_Base_Anes'] = Dist_bet_norm_base
    toplot['BaseAnesnorm_B_Dist_Base_Anes'] = Dist_bet_norm_base_anes

    toplot['W_Dis_Base'] = Dist_w_Base_raw
    toplot['Norm_W_Dis_Base'] = Dist_w_Base_norm
    toplot['W_Dis_Anes'] = Dist_w_Anes_raw
    toplot['Norm_W_Dis_Anes'] = Dist_w_Anes_norm

    toplot.to_csv(f"{output_dir}/DimRed_wPLI.csv", index=False, sep=';')

    # 0 = WSAS non-recovered
    # 1 = WSAS recovered
    # 2 = Healthy

    for i in toplot.columns[2:]:
        plt.figure()
        sns.boxplot(x='outcome', y=i, data=toplot)
        sns.stripplot(x='outcome', y=i, size=4, color=".3", data=toplot)
        plt.xticks([0, 1, 2], ['NonReco', 'Reco', 'Unknown'])
        plt.title(i)
        pdf.savefig()
        plt.close()

    pdf.close()
