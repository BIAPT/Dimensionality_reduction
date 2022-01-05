%% Charlotte Maschke November 11 2020
% This script goal is to generate the time-resolved AEC matrices 
% The matrices will be generated twice: once with overlapping and
% once with non-overlapping windows in the alpha bandwidth.  

%FREQUENCY = "alpha";
%FREQUENCY = "theta";
%FREQUENCY = "delta";
FREQUENCY = "beta";


% Remote Source Setup
%
% Just to try
%INPUT_DIR = 'C:\Users\User\Documents\GitHub\Dimensionality_reduction\data\data_WSAS';

INPUT_DIR = '/home/lotte/projects/def-sblain/lotte/Dim_DOC/data';
OUTPUT_DIR = strcat("/home/lotte/projects/def-sblain/lotte/Dim_DOC/results/", FREQUENCY, "/AEC/");
NEUROALGO_PATH = "/home/lotte/projects/def-sblain/lotte/Dim_DOC/NeuroAlgo";
addpath(genpath(NEUROALGO_PATH)); % Add NA library to our path so that we can use it

% This list contains all participant IDs
P_IDS = {'WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13','WSAS18', 'WSAS19', 'WSAS20', 'WSAS22'};
%CONDITION = {'Base', 'Anes', 'Reco'};
CONDITION = {'Base', 'Anes'};


%% AEC Parameters:
if FREQUENCY == "beta"
    low_frequency = 13;
    high_frequency = 30;
elseif FREQUENCY == "alpha"
    low_frequency = 8;
    high_frequency = 13;
elseif FREQUENCY == "theta"
    low_frequency = 4;
    high_frequency = 8;
elseif FREQUENCY == "delta"
    low_frequency = 1;
    high_frequency = 4;
end

% cut_amount: amount of points from hilbert transform to remove from the start and end.
% the goal is to not keep cut_amount from the start and cut_amount from the end.
cut_amount = 10;
% Size of the cuts for the data
window_size = 10; % in seconds
%this parameter is set to 1 (overlapping windows)and 10(non-overlapping windows).
step_sizes = ["10", "01"]; % in seconds

%% loop over all participants and stepsizes and calculate wPLI
for s = 1:length(step_sizes)
    step = step_sizes{s};
    for p = 1:length(P_IDS)
        p_id = P_IDS{p};

        for c = 1:length(CONDITION)
            cond = CONDITION{c};

            fprintf("Analyzing AEC of participant '%s' with stepsize '%s'  '%s' \n", p_id, step, cond);

            participant_in = strcat(p_id, '_',cond,'_5min.set');
            participant_out_path = strcat(OUTPUT_DIR,'step',step,'/AEC_',FREQUENCY,'_step',step,'_',p_id,'_',cond,'.mat');            
            participant_channel_path = strcat(OUTPUT_DIR,'step',step,'/AEC_',FREQUENCY,'_step',step,'_',p_id,'_',cond,'_channels.mat');            

            %% Load data
            recording = load_set(participant_in,INPUT_DIR);
            sampling_rate = recording.sampling_rate;
            num_channels = recording.number_channels;

            if step == "01"
                step_size = 1;
            elseif step == "10"
                step_size = 10;
            end

            %% Getting the configuration
            configuration = get_configuration();
            frequency_band = [low_frequency high_frequency]; % This is in Hz

            %% Setting Result
            result = Result('aec', recording);
            result.parameters.frequency_band = frequency_band;
            result.parameters.window_size = window_size;
            result.parameters.step_size = step_size;
            result.parameters.cut_amount = cut_amount;

            %% Filtering the data and initiate window
            print_message(strcat("Filtering Data from ",string(frequency_band(1)), "Hz to ", string(frequency_band(2)), "Hz."),configuration.is_verbose);
            recording = recording.filter_data(recording.data, frequency_band);

            % Here we init the sliding window slicing 
            recording = recording.init_sliding_window(window_size, step_size);
            number_window = recording.max_number_window;
            %create windows with filtered data 
            windowed_data = create_sliding_window(recording.filt_data, window_size, step_size, sampling_rate);

            %% initialize empty 3d aec matrix and fill it in a parallized way
            aec_tofill = zeros(number_window, recording.number_channels, recording.number_channels);

            parfor win_i = 1:number_window
                disp(strcat("AEC at window: ",string(win_i)," of ", string(number_window))); 
                segment_data = squeeze(windowed_data(win_i,:,:));
                aec_tofill(win_i,:,:) = aec_pairwise_corrected(segment_data, num_channels, cut_amount);
            end
      
            % correct by taking mean over both diagonals
            % and make matrix symmetric
            aec_tofill = (aec_tofill + permute(aec_tofill,[1,3,2]))/2;

            channels = struct2cell(result.metadata.channels_location);

            %% Save AEC
            save(participant_out_path,'aec_tofill')
            save(participant_channel_path,'channels')
        end
    end
end
