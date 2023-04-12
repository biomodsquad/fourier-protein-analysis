#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# clustermap

def clustermap(data_cm, FeatRanks, comp, xlabel, ylabel) :
    
    data_cm = np.array(data_cm)
    data_cm_trans = data_cm.transpose()
    scaler_cm = StandardScaler()
    scaler_cm.fit(data_cm_trans)
    data_cm_trans_scaled = scaler_cm.transform(data_cm_trans)

    xticklabel = FeatRanks['Allele']
    forytick = []
    for k in range(comp) :
        forytick.append('{}'.format(k+1))
    yticklabel = forytick

    fig, ax = plt.subplots(figsize=(20,5))

    FreqComp = sns.heatmap(data_cm_trans_scaled, xticklabels = xticklabel, yticklabels = yticklabel, annot = False, linewidth = 0)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    FreqComp.set_xticklabels(FreqComp.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    FreqComp.set_yticklabels(FreqComp.get_yticklabels(), rotation = 0, horizontalalignment = 'right')
    FreqComp.set_ylabel(FreqComp.get_ylabel(), rotation = 0, horizontalalignment = 'right')
    plt.show()

    # Generate ClusterMap 

    FreqCompCluster = sns.clustermap(data_cm_trans_scaled, xticklabels = xticklabel, yticklabels= yticklabel, figsize=(15,8))
    plt.setp(FreqCompCluster.ax_heatmap.get_xticklabels(), rotation=45, horizontalalignment = 'right')
    plt.setp(FreqCompCluster.ax_heatmap.get_yticklabels(), rotation=0, horizontalalignment = 'left')
    FreqCompCluster.cax.set_visible(False)
    plt.show()
    
    return 

