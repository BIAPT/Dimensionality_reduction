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
    output_dir = os.path.join(args.output_dir, 'wPLI_Plots')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare output pdf
    pdf = PdfPages(f"{output_dir}/wPLI_Plots_{args.frequencyband}.pdf")

    # load patient IDS
    info = pd.read_csv(args.participants, sep='\t')
    P_IDS = info['Patient']

    for p_id in P_IDS:
        """
        1)    IMPORT DATA
        """
        wpli_Base = np.load(f"{args.input_dir}/wPLI_{args.frequencyband}/WPLI_{args.frequencyband}_{p_id}_Base.npy")
        wpli_Anes = np.load(f"{args.input_dir}/wPLI_{args.frequencyband}/WPLI_{args.frequencyband}_{p_id}_Anes.npy")

        average_wpli_Base = np.mean(wpli_Base, axis=0)
        average_wpli_Anes = np.mean(wpli_Anes, axis=0)

        """
        2)    Plot DATA
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # plot time-averaged wPLI in PDF
        sns.heatmap(average_wpli_Base, cmap='jet', ax=axs[0], vmin=0, vmax=0.25)
        sns.heatmap(average_wpli_Anes, cmap='jet', ax=axs[1], vmin=0, vmax=0.25)
        axs[0].set_title(f"WPLI {args.frequencyband} {p_id} Base")
        axs[1].set_title(f"WPLI {args.frequencyband} {p_id} Anes")
        pdf.savefig(fig)

    pdf.close()
