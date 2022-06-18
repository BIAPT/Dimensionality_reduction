import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
from sklearn.metrics import accuracy_score
from utils import visualize

mode =  "both"

if mode == "both":
    data_w=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('wPLI'))
    data_w=data_w[data_w['ID']!="003MG"]
    data_d=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format('dPLI'))
    data_d=data_d[data_d['ID']!="003MG"]
    areas=data_w.columns[4:]
    info = data_w.iloc[:,:4]
    X = np.hstack((data_w[areas],data_d[areas]))

else:
    data=pd.read_pickle('data/features/23_Part_{}_10_01_alpha.pickle'.format(mode))
    data=data[data['ID']!="003MG"]
    areas=data.columns[4:]
    info = data.iloc[:,:4]
    X = data[areas]

label = np.zeros(len(info['Phase']))
label[np.where(info['Phase']=='Anes')]=1
label[np.where(info['Phase']=='Sedon1')]=1
label[np.where(info['Phase']=='emergencefirst5')]=1

participants = ['WSAS02', 'WSAS05', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
         'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22',
         'MCD0004', 'MCD0007', 'MCD0008', 'MCD0009', 'MCD0012', 'MCD0013','MCD0014', 'MCD0018', 'MCD0021',
         '002MG', '004MG', '004MW']

outcome = ['0', '1', '0', '1', '1', '1', '1', '1', '0', '0', '1',
         '2', '2', '2', '2', '2', '2','2','2','2',
         '3', '3', '3']

pdf = matplotlib.backends.backend_pdf.PdfPages("individual_LDA_BA_{}.pdf".format(mode))

acc = []
prob = []
oneD = []

for part in participants:
    if part.__contains__('MW') or part.__contains__('MG'):
        phases = ['Sedoff', 'Sedon1']
    if part.__contains__('WSAS'):
        phases = ['Base', 'Anes']
    if part.__contains__('MCD'):
        phases = ['eyesclosed1', 'emergencefirst5']

    X_part = X[info['ID'] == part]
    label_part = label[info['ID'] == part]

    # Perform LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_part, label_part)
    label_pred = clf.predict(X_part)
    acc.append(accuracy_score(label_part, label_pred))

    plt.figure()
    plt.plot(clf.predict_proba(X_part))
    plt.title("prediction probability_ " + part)
    pdf.savefig()
    plt.close()

    # plot_feature_weights_brain
    weights = np.array(clf.coef_[0])

    if mode == "both":
        weights_wpli = pd.DataFrame(weights[:55].reshape(-1, len(weights[:55])),columns=areas )
        weights_dpli = pd.DataFrame(weights[55:].reshape(-1, len(weights[55:])),columns=areas )

        visualize.plot_connectivity(weights_wpli)
        plt.title(part + "  wpli")
        pdf.savefig()
        plt.close()

        visualize.plot_connectivity(weights_dpli)
        plt.title(part + " dpli")
        pdf.savefig()
        plt.close()

    else:
        weights = pd.DataFrame(weights.reshape(-1, len(weights)), columns=areas)

        visualize.plot_connectivity(weights)
        plt.title(part + "   " + mode)
        pdf.savefig()
        plt.close()


    X_part_1 = LinearDiscriminantAnalysis(n_components=1).fit(X_part, label_part).transform(X_part)

    data_one = pd.DataFrame(np.transpose(np.vstack((X_part_1.flatten(), label_part))))
    data_one.columns = ['LDA_1D', 'label']
    oneD.append(data_one)


for i,p in enumerate(participants):
    data = oneD[i]
    a = sns.displot(data=data, x = "LDA_1D", hue = "label", kind = "kde",fill=True)
    a._legend.set_title(p)
    pdf.savefig()
    plt.close()


toplot = pd.DataFrame()
toplot['accuracies'] = acc
toplot['ID'] = participants
toplot['outcome'] = outcome

plt.figure()
sns.boxplot(x = 'outcome', y = 'accuracies', data = toplot)
sns.stripplot(x = 'outcome', y = 'accuracies', size=4, color=".3",data = toplot)
plt.xticks([0,1,2,3],['recovered','non-recoveded','healthy','NET_ICU'])
plt.title("LDA_training_accuracy")
#plt.ylim(0.98,1.02)
pdf.savefig()
plt.close()

pdf.close()

