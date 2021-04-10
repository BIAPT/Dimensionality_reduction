import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import accuracy_score
from utils import visualize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.model_selection import cross_validate

mode = "dPLI"
#mode = "wPLI"
#mode = "both"


if mode == "both":
    data_w=pd.read_pickle('../data/features/23_Part_{}_10_01_alpha.pickle'.format('wPLI'))
    data_w=data_w[data_w['ID']!="003MG"]
    data_d=pd.read_pickle('../data/features/23_Part_{}_10_01_alpha.pickle'.format('dPLI'))
    data_d=data_d[data_d['ID']!="003MG"]
    areas=data_w.columns[4:]
    info = data_w.iloc[:,:4]
    X = np.hstack((data_w[areas],data_d[areas]))

else:
    data=pd.read_pickle('../data/features/23_Part_{}_10_01_alpha.pickle'.format(mode))
    data=data[data['ID']!="003MG"]
    areas=data.columns[4:]
    info = data.iloc[:,:4]
    X = data[areas]

label = np.zeros(len(info['Phase']))
label[np.where(info['Phase']=='Anes')]=1
label[np.where(info['Phase']=='Sedon1')]=1
label[np.where(info['Phase']=='emergencefirst5')]=1

participants = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13', 'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22']

outcome = ['0', '1', '0', '1', '1', '1', '1', '1', '0', '0', '1']

pdf = matplotlib.backends.backend_pdf.PdfPages("WSAS_individual_SVM_PCA_BA_{}.pdf".format(mode))

SVM_weights = []
SVM_acc = []
oneD_Dist = []
oneD_PCA = []
oneD_var = []
oneD_varrat = []
allD_Dist_w_Anes  = []
allD_Dist_w_Base = []
allD_Dist_bet = []
allD_variance = []
cross_val_acc =[]
allD_Dist_bet_norm = []
allD_Dist_bet_norm2 = []

for part in participants:
    if part.__contains__('MW') or part.__contains__('MG'):
        phases = ['Sedoff', 'Sedon1']
    if part.__contains__('WSAS'):
        phases = ['Base', 'Anes']
    if part.__contains__('MCD'):
        phases = ['eyesclosed1', 'emergencefirst5']

    X_part = X[info['ID'] == part]
    label_part = label[info['ID'] == part]

    """
    Run SVM 
    --> training accuracy
    --> plot weights 
    """

    # Perform SVM
    clf = LinearSVC()
    clf.fit(X_part, label_part)
    label_pred = clf.predict(X_part)
    SVM_acc.append(accuracy_score(label_part, label_pred))

    # plot_feature_weights_brain
    weights = np.array(abs(clf.coef_[0]))
    weights = (weights - min(weights)) / (max(weights) - min(weights))

    SVM_weights.append(weights)

    if mode == "both":
        weights_wpli = pd.DataFrame(weights[:55].reshape(-1, len(weights[:55])),columns=areas )
        weights_dpli = pd.DataFrame(weights[55:].reshape(-1, len(weights[55:])),columns=areas )

        visualize.plot_connectivity(weights_wpli)
        plt.text(-20, 0.75, part + "  wpli")
        pdf.savefig()
        plt.close()

        visualize.plot_connectivity(weights_dpli)
        plt.text(-20, 0.75, part + "  dpli")
        pdf.savefig()
        plt.close()

    else:
        weights = pd.DataFrame(weights.reshape(-1, len(weights)), columns=areas)

        visualize.plot_connectivity(weights)
        plt.text(-20, 0.75, part + "   " + mode)
        pdf.savefig()
        plt.close()

    """
    Run SVM 
    --> cv-accuracy
    """
    # Perform SVM
    clf = LinearSVC()
    scores = cross_validate(clf, X_part, label_part, cv=10)
    cross_val_acc.append(np.mean(scores['test_score']))

    """
    Get High Dimensional Distance
    
    overall_dist = np.mean(distance.cdist(X_part, X_part, 'euclidean'))

    dist_bet = distance.cdist(X_part[label_part==0], X_part[label_part==1], 'euclidean')
    allD_Dist_bet.append(np.mean(dist_bet))

    allD_Dist_bet_norm.append(np.mean(dist_bet)/overall_dist)

    dist_within_b = distance.cdist(X_part[label_part==0], X_part[label_part==0], 'euclidean')
    allD_Dist_w_Base.append(np.mean(dist_within_b))

    dist_within_a = distance.cdist(X_part[label_part==1], X_part[label_part==1], 'euclidean')
    allD_Dist_w_Anes.append(np.mean(dist_within_a))

    allD_Dist_bet_norm2.append(np.mean(dist_bet) / np.mean(dist_within_b))
    """

    """
        Get High Dimensional Distance
    """
    overall_dist = np.sum(distance.cdist(X_part, X_part, 'euclidean'))

    dist_bet = np.sum(distance.cdist(X_part[label_part==0], X_part[label_part==1], 'euclidean'))/overall_dist
    allD_Dist_bet.append(dist_bet)

    dist_within_b = np.sum(distance.cdist(X_part[label_part==0], X_part[label_part==0], 'euclidean'))/overall_dist
    allD_Dist_w_Base.append(dist_within_b)

    dist_within_a = np.sum(distance.cdist(X_part[label_part==1], X_part[label_part==1], 'euclidean'))/overall_dist
    allD_Dist_w_Anes.append(dist_within_a)

    """
    Run PCA 1D
    --> Distance
    --> explained Variance 
    """
    pca_1 = PCA(n_components=1).fit(X_part)
    X_part_1 = pca_1.transform(X_part)
    data_one = pd.DataFrame(np.transpose(np.vstack((X_part_1.flatten(), label_part))))
    data_one.columns = ['PCA_1D', 'label']
    oneD_PCA.append(data_one)

    mean_base = np.mean(data_one[label_part == 0]['PCA_1D'])
    mean_anes = np.mean(data_one[label_part == 1]['PCA_1D'])
    dist_tmp = max(mean_base, mean_anes) - min(mean_base, mean_anes)
    oneD_Dist.append(dist_tmp)

    oneD_varrat.append(pca_1.explained_variance_ratio_[0])
    oneD_var.append(pca_1.explained_variance_[0])


    """
    visualize data on 2D
    """

    X_part_2 = PCA(n_components=2).fit(X_part).transform(X_part)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(X_part_2[label_part == 0, 0], X_part_2[label_part == 0, 1], 'red')
    ax.plot3D(X_part_2[label_part == 1, 0], X_part_2[label_part == 1, 1], 'blue')
    plt.title(part)
    pdf.savefig()
    plt.close()

    """
    get overall variance in data
    """

    pca = PCA().fit(X_part)
    allD_variance.append(sum(pca.explained_variance_))



for i,p in enumerate(participants):
    data = oneD_PCA[i]
    a = sns.displot(data=data, x = "PCA_1D", hue = "label", kind = "kde",fill=True)
    a._legend.set_title(p)
    pdf.savefig()
    plt.close()

if mode == 'both':
    SVM_weights = pd.DataFrame(SVM_weights)
    nonr_weight = SVM_weights[np.array(outcome) == '1']
    reco_weight = SVM_weights[np.array(outcome) == '0']

if mode != 'both':
    SVM_weights = pd.DataFrame(SVM_weights, columns=areas)
    nonr_weight = SVM_weights[np.array(outcome) == '1']
    reco_weight = SVM_weights[np.array(outcome) == '0']


    for i in range(0,len(nonr_weight)):
        tmp  = np.array(nonr_weight.iloc[i,:])
        weights_tmp = pd.DataFrame(tmp.reshape(-1, len(tmp)), columns=areas)
        visualize.plot_connectivity(weights_tmp)
        plt.text(-20,0.75,"SVM_non-recovered")
        pdf.savefig()
        plt.close()

    for i in range(0,len(reco_weight)):
        tmp  = np.array(reco_weight.iloc[i,:])
        weights_tmp = pd.DataFrame(tmp.reshape(-1, len(tmp)), columns=areas)
        visualize.plot_connectivity(weights_tmp)
        plt.text(-20,0.75, "SVM_recovered")
        pdf.savefig()
        plt.close()

cosine_non = distance.cdist(nonr_weight, nonr_weight, 'cosine').reshape(-1)
cosine_reco = distance.cdist(reco_weight, reco_weight, 'cosine').reshape(-1)
cosine_nonr_reco = distance.cdist(reco_weight, nonr_weight, 'cosine').reshape(-1)

plt.figure()
plt.violinplot([cosine_non[cosine_non > 0.0001],
                cosine_reco[cosine_reco > 0.0001],
                cosine_nonr_reco[cosine_nonr_reco > 0.0001]])
plt.xticks([1,2,3],['recovered','nonrecovered','between r-n'])
plt.title('Cosine Similarity')
pdf.savefig()
plt.close()

toplot = pd.DataFrame()
toplot['accuracies'] = SVM_acc
toplot['10_fold_cv'] = cross_val_acc
toplot['PCA variance'] = oneD_var
toplot['PCA variance_ratio'] = oneD_varrat
toplot['PCA Distance'] = oneD_Dist
toplot['Distance between Base-Anes'] = allD_Dist_bet
#toplot['Norm_Distance between Base-Anes'] = allD_Dist_bet_norm
#toplot['Norm2_Distance between Base-Anes'] = allD_Dist_bet_norm
toplot['Distance within Anes'] = allD_Dist_w_Anes
toplot['Distance within Base'] = allD_Dist_w_Base
toplot['overall variance'] = allD_variance
toplot['ID'] = participants
toplot['outcome'] = outcome

for i in toplot.columns[:-2]:
    plt.figure()
    sns.boxplot(x = 'outcome', y = i, data = toplot)
    sns.stripplot(x = 'outcome', y = i, size=4, color=".3",data = toplot)
    plt.xticks([0,1],['recovered','non-recoveded'])
    plt.title( i )
    pdf.savefig()
    plt.close()

out_color = [int(numeric_string) for numeric_string in np.array(toplot['outcome'])]


plt.figure()
n = np.where(toplot['outcome']=='0')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n], y = toplot['Distance within Base'][n],
            label = 'recovered')
n = np.where(toplot['outcome']=='1')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n], y = toplot['Distance within Base'][n],
            label = 'non- recovered')
plt.xlabel("bet Base-Anes")
plt.ylabel("with Base")
plt.legend()
pdf.savefig()
plt.close()

plt.figure()
n = np.where(toplot['outcome']=='0')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n],
            y = toplot['Distance within Base'][n]-toplot['Distance within Anes'][n],
            label = 'recovered')
n = np.where(toplot['outcome']=='1')[0]
plt.scatter(x= toplot['Distance between Base-Anes'][n],
            y = toplot['Distance within Base'][n]-toplot['Distance within Anes'][n],
            label = 'non- recovered')
plt.xlabel("bet Base-Anes")
plt.ylabel("with Base-with Anes")
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
plt.xlabel("bet Base-Anes")
plt.ylabel("with Anes")
pdf.savefig()
plt.close()

pdf.close()

