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


frequency = "alpha"
STEP = "01"                     # stepsize: can be "10"
MODE = "wpli"                   # can be "dpli", "wpli", "both"
dist_type = "euclidean"         #"euclidean"


IDS = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
         'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22']
outcome = ['0', '1', '0', '1', '1', '1', '1', '1', '0', '0', '1']


INPUT_DIR = "../data/connectivity/{}/{}/step{}/".format(frequency, MODE, STEP)
pdf = matplotlib.backends.backend_pdf.PdfPages("Unaveraged_Distance_{}.pdf".format(MODE))


allD_variance = []
allD_variance_on = []
allD_variance_off = []
SVM_acc_train = []
SVM_acc_cv = []
Dist_overall = []
Dist_bet = []
allD_Dist_bet_norm =[]
Dist_w_on  = []
Dist_w_off = []
Dist_w_change = []
oneD_Dist = []
oneD_var = []
oneD_varrat = []
oneD_PCA =[]


for p_id in IDS:
    """
    1)    IMPORT DATA
    """

    if p_id.__contains__('MW') or p_id.__contains__('MG'):
            OFF = 'Sedoff'
            ON = 'Sedon1'
    if p_id.__contains__('WSAS'):
            OFF = 'Base'
            ON = 'Anes'
    if p_id.__contains__('MCD'):
            OFF = 'eyesclosed1'
            ON = 'emergencefirst5'

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
    nr_timesteps = min(len(data_off),len(data_off))

    data_on_2d = np.empty((len(data_on),nr_features))
    data_off_2d = np.empty((len(data_off),nr_features))

    for i in range(nr_timesteps):
        data_on_2d[i] = data_on[i][np.triu_indices(len(common_channels)-1)]
        data_off_2d[i] = data_off[i][np.triu_indices(len(common_channels)-1)]

    # combine the data over time
    data_combined = np.vstack((data_off_2d, data_on_2d))

    # define y-value (On/off) to combine data
    Y = np.zeros(len(data_combined))   # 0 is OFF
    Y[0:nr_timesteps] = 1              # 1 is ON

    """
    2)    RUN PCA
    """
    pca_2 = PCA(n_components=2).fit(data_combined)
    data_combined_2 = pca_2.transform(data_combined)

    plt.figure()
    n = np.where(Y==0)
    plt.scatter(data_combined_2[n,0],data_combined_2[n,1],c = 'red',label = 'OFF')
    n = np.where(Y==1)
    plt.scatter(data_combined_2[n,0],data_combined_2[n,1],c = 'blue',label = 'ON')
    plt.legend()
    plt.title(p_id)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    pdf.savefig()
    plt.close()

    """
    Run PCA 1D
    --> Distance
    --> explained Variance 
    """
    pca_1 = PCA(n_components=1).fit(data_combined)
    X_part_1 = pca_1.transform(data_combined)
    data_one = pd.DataFrame(np.transpose(np.vstack((X_part_1.flatten(), Y))))
    data_one.columns = ['PCA_1D', 'label']
    oneD_PCA.append(data_one)

    mean_off = np.mean(data_one[Y == 0]['PCA_1D'])
    mean_on = np.mean(data_one[Y == 1]['PCA_1D'])
    dist_tmp = max(mean_off, mean_on) - min(mean_off, mean_on)
    oneD_Dist.append(dist_tmp)

    oneD_varrat.append(pca_1.explained_variance_ratio_[0])
    oneD_var.append(pca_1.explained_variance_[0])


    """
    Reduce data to 80%
    
    # fit pca and only keep 80% variance
    pca = PCA().fit(data_combined)

    allD_variance.append(sum(pca.explained_variance_))

    nr_PC = np.min(np.where(np.cumsum(pca.explained_variance_ratio_)>0.80)[0])
    if nr_PC == 0:
        nr_PC = 1
    print(str(nr_PC))
    data_combined = PCA(n_components=nr_PC ).fit_transform(data_combined)
    

    Run SVM 
    --> training accuracy
    --> plot weights 
    
    # Perform SVM
    clf = LinearSVC(max_iter=100000)
    clf.fit(data_combined, Y)
    y_pred = clf.predict(data_combined)
    SVM_acc_train.append(accuracy_score(Y, y_pred))

        Run SVM 
        --> cv-accuracy
    # Perform SVM
    clf = LinearSVC(max_iter=100000)
    scores = cross_validate(clf, data_combined, Y, cv=20)
    SVM_acc_cv.append(np.mean(scores['test_score']))
    """

    """
        3)    CALCULATE DISTANCES
    """
    pca = PCA().fit(data_combined)
    allD_variance.append(sum(pca.explained_variance_))

    pca = PCA().fit(data_combined[Y==0])
    allD_variance_off.append(sum(pca.explained_variance_))

    pca = PCA().fit(data_combined[Y==1])
    allD_variance_on.append(sum(pca.explained_variance_))

    data_combined = PCA(n_components=60).fit_transform(data_combined)

    # distance over all points
    overall_dist = np.sum(distance.cdist(data_combined, data_combined, dist_type))
    Dist_overall.append(overall_dist)

    # distance between ON and OFF
    dist_bet = np.sum(distance.cdist(data_combined[Y==0], data_combined[Y==1], dist_type))/overall_dist
    Dist_bet.append(dist_bet)

    # distance within ON or OFF
    dist_within_on = np.sum(distance.cdist(data_combined[Y==1], data_combined[Y==1], dist_type))/overall_dist
    dist_within_off = np.sum(distance.cdist(data_combined[Y==0], data_combined[Y==0], dist_type))/overall_dist
    Dist_w_on.append(dist_within_on)
    Dist_w_off.append(dist_within_off)

    Dist_w_change.append(max(dist_within_off,dist_within_on)-min(dist_within_off,dist_within_on))

for i,p in enumerate(IDS):
    data = oneD_PCA[i]
    a = sns.displot(data=data, x = "PCA_1D", hue = "label", kind = "kde",fill=True)
    a._legend.set_title(p)
    pdf.savefig()
    plt.close()


toplot = pd.DataFrame()
toplot['ID'] = IDS
toplot['outcome'] = outcome
#toplot['Variance in data'] = SVM_acc_train
#toplot['SVM_train'] = SVM_acc_train
#toplot['SVM_CV'] = SVM_acc_cv
#toplot['one_Dim_Var_ratio'] = oneD_varrat
#toplot['One Dim var'] = oneD_var
toplot['All Dim var'] = allD_variance
toplot['One Dim Dist'] = oneD_Dist
toplot['Overall Distance'] = Dist_overall
toplot['Distance between Base-Anes'] = Dist_bet
toplot['Distance within Anes'] = Dist_w_on
toplot['Distance within Base'] = Dist_w_off
toplot['Distance within Change'] = Dist_w_change


for i in toplot.columns[2:]:
    plt.figure()
    sns.boxplot(x = 'outcome', y = i, data = toplot)
    sns.stripplot(x = 'outcome', y = i, size=4, color=".3",data = toplot)
    plt.xticks([0,1],['recovered','non-recoveded'])
    plt.title( i )
    pdf.savefig()
    plt.close()

# do sone nice scatterplots for the group

plt.figure()
n = np.where(toplot['outcome']=='0')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n], y = toplot['Distance within Base'][n],
            label = 'recovered')
n = np.where(toplot['outcome']=='1')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n], y = toplot['Distance within Base'][n],
            label = 'non- recovered')
plt.xlabel("Distance between Base-Anes")
plt.ylabel("Distance within Base")
plt.legend()
pdf.savefig()
plt.close()

plt.figure()
n = np.where(toplot['outcome']=='0')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n], y = toplot['Distance within Anes'][n],
            label = 'recovered')
n = np.where(toplot['outcome']=='1')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n], y = toplot['Distance within Anes'][n],
            label = 'non-recovered')
plt.legend()
plt.xlabel("Distance between Base-Anes")
plt.ylabel("Distance with Anes")
pdf.savefig()
plt.close()

plt.figure()
n = np.where(toplot['outcome']=='0')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n], y = toplot['Distance within Change'][n],
            label = 'recovered')
n = np.where(toplot['outcome']=='1')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n], y = toplot['Distance within Change'][n],
            label = 'non-recovered')
plt.legend()
plt.xlabel("Distance between Base-Anes")
plt.ylabel("Change in Distance with")
pdf.savefig()
plt.close()



pdf.close()



