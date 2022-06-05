#!/usr/bin/env python
import os
import mne
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from src.BIAPT_Connectivity import connectivity_compute

FREQUENCIES  = {
  "delta": (1,4),
  "theta": (4,8),
  "alpha": (8,13),
  "beta": (13,30),
  "gamma": (30,45),
  "fullband": (30,45),
  }

WINDOW_LENGTH = 10
STEP_SIZE = 1
N_SURROGATES = 20

def data_import(input_dir, p_id, cond):
    # define epoch name
    input_fname =  f"{input_dir}/sub-{p_id}/eeg/epochs_{p_id}_{cond}.fif"
    raw_epochs = mne.read_epochs(input_fname)

    # remove channels marked as bad and non-brain channels
    raw_epochs.drop_channels(raw_epochs.info['bads'])

    #crop data if necessary
    if len(raw_epochs) > 30:
        epochs_cropped = raw_epochs[-30:]
    else:
        epochs_cropped = raw_epochs.copy()

    return epochs_cropped


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calculates wPLI functional connectivity.')
    parser.add_argument('-input_dir', type=str,
                        help='folder name containing the data in epoched .fif data in BIDS format')
    parser.add_argument('-output_dir', type=str,
                        help='folder name where to save the functional connectivity')
    parser.add_argument('-participants', type=str,
                        help='path to txt with information about participants')
    parser.add_argument('-frequencyband', type=str,
                        help='lower and upper filer frequency')
    parser.add_argument('-id', type=str,# default = None,
                        help='Participant ID to compute')
    args = parser.parse_args()

    ################################################################
    #       1)    PREPARE IN-AND OUTPUT                            #
    ################################################################

    # make ouput directory
    output_dir = os.path.join(args.output_dir, f'wPLI_{args.frequencyband}')
    os.makedirs(output_dir, exist_ok=True)

    # prepare output pdf
    pdf = PdfPages(os.path.join(output_dir, f"functional_connectivity_{args.frequencyband}.pdf"))

    # load patient IDS
    info = pd.read_csv(args.participants, sep='\t')
    P_IDS = [info['Patient'][args.id]] if args.id else info['Patient']

    l_freq, h_freq = FREQUENCIES[args.frequencyband]

    for p_id in P_IDS:

        # import data from both conditions
        epochs_Base = data_import(args.input_dir, p_id, cond="Base")
        epochs_Anes = data_import(args.input_dir, p_id, cond="Anes")

        # find channels that exist in both datasets and drop others
        intersect = list(np.intersect1d(epochs_Base.info['ch_names'], epochs_Anes.info['ch_names']))
        drop_A = set(epochs_Anes.info['ch_names']) ^ set(intersect)
        drop_B = set(epochs_Base.info['ch_names']) ^ set(intersect)

        epochs_Anes.drop_channels(drop_A)
        Anes = np.concatenate(list(epochs_Anes), axis=1)

        epochs_Base.drop_channels(drop_B)
        Base = np.concatenate(list(epochs_Base), axis=1)

        sfreq = int(epochs_Base.info['sfreq'])

        #epochs_Anes = np.array(epochs_Anes)
        #epochs_Base = np.array(epochs_Base)

        arguments = (WINDOW_LENGTH, STEP_SIZE, l_freq, h_freq, sfreq)
        kwargs = {"mode": "wpli", "verbose": True, "n_surrogates": N_SURROGATES}

        wpli_Base = connectivity_compute(Base, *arguments, **kwargs)
        wpli_Anes = connectivity_compute(Anes, *arguments, **kwargs)

        average_wpli_Base = np.mean(wpli_Base, axis=0)
        average_wpli_Anes = np.mean(wpli_Anes, axis=0)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # plot time-averaged wPLI in PDF
        sns.heatmap(average_wpli_Base, cmap='jet', ax=axs[0], vmin=0, vmax=0.25)
        sns.heatmap(average_wpli_Anes, cmap='jet', ax=axs[1], vmin=0, vmax=0.25)
        axs[0].set_title(f"WPLI 4-8 Hz {p_id} Base")
        axs[1].set_title(f"WPLI 4-8 Hz {p_id} Anes")
        pdf.savefig(fig)

        np.save(os.path.join(output_dir, f"WPLI_{args.frequencyband}_{p_id}_Base.npy"), wpli_Base)
        np.save(os.path.join(output_dir, f"WPLI_{args.frequencyband}_{p_id}_Anes.npy"), wpli_Anes)

    pdf.close()
