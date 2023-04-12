#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages/modules
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt   
import seaborn as sns             
from matplotlib.backends.backend_pdf import PdfPages
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# Import functions
from kernelfunc import KN
from featsel import featSelect
from heatmap import heatmap
from svmtune import tuneSVM
from clustermap import clustermap


# In[2]:


## Assign input variables

# classification cutoff (nM)
cutoff = 1000

# imbalance threshold (%)
imb_threshold = 35  

# number of training sets
NT = 5 

# hyper-parameter ranges
C_range = np.logspace(start= -3, stop = 3, num=7, endpoint= True, base = 10).tolist()
gamma_range = np.logspace(start= -3, stop = 3, num=7, endpoint= True, base = 10).tolist()

# data split ratio
blind_size = 0.2
val_size = 0.2

#reduction factor
red_factor = 1/3

# interval for feature selection validation
intv = 10

#sampling method (SMOTE(), ADASYN)
sampler = SMOTE()


# In[3]:


# file inputs
filename_NumRep = "blosum100.csv"
filename_pepDB = "pHLA_data_20230206.csv"


# In[4]:


### Notes for users
# NumRep consists of only numbers. Each row is a numeric representation of an amino acid.


# In[5]:


# Import amino acids numeric representation file
NumRep = pd.read_csv(filename_NumRep, sep = ",", header = None)  
AA = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
NumRepAA = NumRep.copy()
NumRepAA['AA'] = AA

NumRepAA


# In[6]:


# Import peptide database
pepDB = pd.read_csv(filename_pepDB, sep = ",", header = 0) 

# Assign the index numbers for all entries
pepDB['index'] = pepDB.index

# Replace *-:/ with _
trouble = ['*',"-",":","/"]
for i in trouble :
    pepDB['Allele'] = pepDB['Allele'].str.replace(i,"_") 

pepDB


# In[7]:


# Get Maximum length of peptides
maxlen = pepDB['Length'].max()

# Determine signal length
power = np.ceil(np.log2(maxlen))
siglen =int(2**(power))

LenPep = int(siglen/2+1)
LenRep = NumRep.shape[1]
LenFeats =LenPep*LenRep

# Get frequency vector
freq = np.linspace(-0.5,0.5, num= siglen+1, endpoint = True)

print("Maximum Length of peptides : {}".format(maxlen))
print("Signal Length : {}".format(siglen))
print("Frequency vector : {}".format(freq))


# In[8]:


# Encode peptide sequences using Fast Fourier Transform 

c = 0
fTab = np.zeros((len(pepDB['Sequence']),int((siglen/2+1)*LenRep)))   

for pept in pepDB['Sequence'] :
    p = list(pept)  
    nTab = np.zeros((len(p), LenRep)) 
    
    for i in range(0,len(p)) :
        if p[i] in AA :    
            nTabr = NumRepAA[NumRepAA['AA'].str.contains(p[i],case=False)].drop('AA', axis =1).to_numpy() 
            nTab[i,:] = nTabr  
         
    feats = []  
    N = len(pept) 
    
    for nn in range(0,LenRep) :
        tmp = [0]*(siglen)
        
        for j in range(0,N) :
            tmp[j] = nTab[j,nn]  
    
        # fast fourier transform
        FT = fft(tmp)  
        FT = np.append(FT,FT[0])  
    
        feats_tmp = []
        
        for k in range(int(siglen/2),int(siglen+1)) :  
            feats_tmp = np.append(feats_tmp, abs(FT[k])/N)  
                                                            
        feats = np.append(feats, feats_tmp)
    
    fTab[c,:] = feats  
    c += 1   




fTab = pd.DataFrame(fTab)
fTab.to_csv('fTab.csv', index = False)

fTab
#plt.plot(freq, FT)
#plt.plot(freq, abs(FT))


# In[9]:


fTab = pd.read_csv("fTab.csv", sep    = ",", header = 0)

# Assign lables based on a cutoff (binary classification)
trainInput = pd.DataFrame(fTab)
Class = []

for i in range(0,len(pepDB)) :
    if pepDB.loc[i].at["BA"] > cutoff :
        Class.append('NB')
    else :
        Class.append('B')

# get trainInput for training and pepDB for statistics
trainInput['Class'] = Class
trainInput['index'] = pepDB['index']
trainInput['Allele'] = pepDB['Allele']
pepDB['Class'] = trainInput['Class']

pepDB


# In[10]:


trainInput


# In[11]:


# Scale input X


X = trainInput.iloc[:,0:LenFeats]

scaler = StandardScaler()
scaler.fit(X)
X_scaled_all = scaler.transform(X)

trainInputScaled = pd.concat([ pd.DataFrame(X_scaled_all), trainInput['Class'], trainInput['index'],trainInput['Allele']], axis=1)

trainInputScaled


# In[12]:


# Get dataframes by alleles

Allele_list = trainInputScaled['Allele'].drop_duplicates().tolist()
NumAllele = len(Allele_list)
AS_tIS = {}
for Alleles in Allele_list :
    AS_tIS[Alleles]=trainInputScaled[trainInputScaled['Allele'].str.match(Alleles)]
    
Allele_list


# In[13]:


AS_tIS['HLA_DRB1_04_01']


# In[14]:


# Create dataframes for statistics

CL0 = ['Alleles', 'N_total', 'N_B', 'N_NB', 'P_B', 'P_NB']
DataStats = pd.DataFrame(columns = CL0)

CL1 = ['Allele'] + list(range(1,LenFeats+1))
FeatRanks = pd.DataFrame(columns = CL1)

numFeat = []
for i in range(intv,LenFeats+1,intv) :
    numFeat.append(i)
CL2 = ['Allele'] + numFeat
FeatSetAccu = pd.DataFrame(columns = CL2)

CL3 = ['Allele', 'Number of features used']
NumFeatUsed = pd.DataFrame(columns = CL3)

CL4 = ['Allele', 'best C', 'best gamma', 'Max AUC' ]
bp_max_auc = pd.DataFrame(columns = CL4)

CL5 = ['Allele', 'Train AUC', 'Prediction AUC']
TrainAUC_BlindAUC = pd.DataFrame(columns = CL5)


# In[ ]:


### SVM

X_Blind_Sets = {}
y_Blind_Sets = {}
AS_cgauc = {}

heatmapGS = PdfPages('Heatmaps_GridSearch.pdf')
featselplot = PdfPages('FeatureSelectionPlot.pdf')
ROCcurve_AUC = PdfPages('ROCcurve_AUC')

for Alleles in Allele_list :
    
    ## data preparation for training and cross-validation
    
    # get allele-specific dataset
    dataset_SVM = AS_tIS[Alleles] 
    X_scaled = dataset_SVM.iloc[:,0:LenFeats]
    y = dataset_SVM['Class']
    
    # save data statistics
    N_total = len(dataset_SVM)
    N_B = dataset_SVM['Class'].str.match('B').sum()
    N_NB = dataset_SVM['Class'].str.match('NB').sum()
    P_B = N_B/N_total*100
    P_NB = N_NB/N_total*100

    DataStats.loc[len(DataStats.index)] = [Alleles, N_total, N_B, N_NB, P_B, P_NB]
    
    # OverSampling/SMOTE for imbalanced data
    if P_B < imb_threshold or P_NB < imb_threshold :
        X_scaled, y = sampler.fit_resample(X_scaled, y)
        
    # split dataset into train set and blind set
    X_train_all, X_blind, y_train_all, y_blind = train_test_split(X_scaled, y,stratify =y, random_state= None, test_size= blind_size)

    X_Blind_Sets[Alleles] = X_blind 
    y_Blind_Sets[Alleles] = y_blind
    
    # split train set into train and test(val) data / repeat NT times and |get NT number of datsets
    N_test = int(np.ceil(len(X_train_all)*0.2))
    X_train_sets = np.empty((NT,len(X_train_all)-N_test,LenFeats))
    X_test_sets = np.empty((NT,N_test,LenFeats))      
    y_train_sets =  [[] for i in range(NT)]   
    y_test_sets =[[] for i in range(NT)]   

    for i in range(0,NT) :
        X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, stratify = y_train_all, random_state= None, test_size= val_size)
        X_train_sets[i,:,:] = X_train
        X_test_sets[i,:,:]  = X_test
        y_train_sets[i] = y_train
        y_test_sets[i] = y_test


    ## feature selection

    # generate feature rank using function featSelect
    featRank = featSelect(X_train_all, y_train_all, X_train_sets, y_train_sets, X_test_sets, y_test_sets, LenFeats, C_range, gamma_range, redF = red_factor)
    
    sf =  np.argsort(featRank).mean(axis=0).argsort() 
    
    allelesf = [Alleles] + sf.tolist()
    FeatRanks.loc[len(FeatRanks.index)] = allelesf

    # Validation of feature selection
    featAccu = []
    nFeat = []

    for i in range(intv, LenFeats +1,intv) :      
        selFeat = sf[0:i]
        tunePer = tuneSVM(X_train_all, y_train_all, X_train_sets[:,:,selFeat], y_train_sets, X_test_sets[:,:,selFeat], y_test_sets, costs= C_range, gammas = gamma_range)
        nFeat.append(i)
        featAccu.append(tunePer[:,2].max())
    
    AlleFeatAccu = [Alleles] + featAccu
    FeatSetAccu.loc[len(FeatSetAccu.index)] = AlleFeatAccu
    
    # feature selection plot
    fig, ax = plt.subplots(1,1, figsize=(7, 4))
    plt.plot(nFeat, featAccu) 
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Number of features')
    plt.title('Validation Accuracy with sets of features  '+'['+Alleles+']')
    plt.show()

    featselplot.savefig(fig)
    ## train SVM model with the selected features
    
    # Prepare data with the selected features
    NumFeat = nFeat[featAccu.index(max(featAccu))]  
    feats_id = sf[0:NumFeat] 

    NumFeatUsed.loc[len(NumFeatUsed.index)] = NumFeat
    
    X_train_all = X_train_all.to_numpy()
    X_blind = X_blind.to_numpy()

    X_train_all_sf = X_train_all[:, feats_id]
    y_train_all_sf = y_train_all

    X_train_sets_sf = X_train_sets[:,:, feats_id]
    y_train_sets_sf = y_train_sets
    X_test_sets_sf = X_test_sets[:,:, feats_id]
    y_test_sets_sf = y_test_sets

    X_blind_sf = X_blind[:, feats_id]
    y_blind_sf = y_blind

    # grid search
    tunePer_sf = tuneSVM(X_train_all_sf, y_train_all_sf, X_train_sets_sf, y_train_sets_sf, X_test_sets_sf, y_test_sets_sf, costs= C_range, gammas = gamma_range)
    AS_cgauc[Alleles]= tunePer_sf  # save cgaccu
    
    # heatmaps
    aucs = np.zeros(((len(C_range), len(gamma_range))))
    for i in range(0,len(C_range)) :
        aucs[i,:] = tunePer_sf[i*7:(i+1)*7,2]
    heatmap(heatmapGS, aucs, Alleles, C_range, gamma_range)
    

    # best parameters from grid search        
    max_auc_id = tunePer_sf[:,2].argmax(axis = 0)
    bp_c = tunePer_sf[max_auc_id, 0]
    bp_g = tunePer_sf[max_auc_id, 1]
    max_auc = tunePer_sf[max_auc_id,2]

    bp_max_auc.loc[len(bp_max_auc.index)] = [Alleles, bp_c, bp_g, max_auc]   # save best params and max accuracy


    # fit SVM
    svc = SVC(kernel = 'rbf', C =bp_c, gamma =bp_g).fit(X_train_all_sf, y_train_all_sf)
    dump(svc, 'SVM_{}.joblib'.format(Alleles)) # save SVMs / Use when reload [clf = load('SVM_{}.joblib'.format(Alleles))] 
    
    print("MHC Allele : {}".format(Alleles))
    
    auc_train = roc_auc_score(y_train_all_sf, svc.decision_function(X_train_all_sf))
    print("Training AUC : {}".format(auc_train))

    # validate the SVM with blind test set
    auc_blind = roc_auc_score(y_blind_sf, svc.decision_function(X_blind_sf))
    print("Blind Test AUC : {}".format(auc_blind))
    
    # ROC Curve
    svc_disp = RocCurveDisplay.from_estimator(svc, X_blind_sf, y_blind_sf, pos_label ='B')
    ROCcurve = svc_disp
    plt.show()
    #ROCcurve_AUC.savefig(ROCcurve)
    
    TrainAUC_BlindAUC.loc[len(TrainAUC_BlindAUC.index)] = [Alleles, auc_train, auc_blind]

heatmapGS.close()
featselplot.close()
#ROCcurve_AUC.close()

# save results
DataStats.to_csv('DataStatistics.csv', index = False)
FeatRanks.to_csv('FeatureRanks.csv', index = False) # from 0 to 169
FeatSetAccu.to_csv('FeatureSetsAccuracy.csv', index = False)
NumFeatUsed.to_csv('NumberOfFeaturesUsed.csv',index = False)
#AS_cgaccu.to_csv('C_gamma_Accuracy_TuneSVM.csv', index = False)  # dict to csv how ????
bp_max_auc.to_csv('BestParams(C,g)_MaxAccuracy_TuneSVM.csv', index = False) # merge these two
TrainAUC_BlindAUC.to_csv('Training_BlindTest_AUC.csv', index = False)  # merge these two


# In[ ]:


DataStats.sort_values(by = 'P_NB')


# In[ ]:


FeatRanks


# In[ ]:


# ClusterMap

FeatRanks_heatmap = pd.DataFrame(FeatRanks)
FeatRanks_heatmap = FeatRanks.drop('Allele', axis=1)


for i in range(len(Allele_list)):
    FeatRanks_heatmap.iloc[i,:] = np.argsort(FeatRanks_heatmap.iloc[i,:])
    FeatRanks_heatmap.iloc[i,:] = FeatRanks_heatmap.iloc[i,:].rank(ascending = False)# the higher the number, the more important the feature
FeatRanks_heatmap.columns = range(FeatRanks_heatmap.columns.size) # reset column index # x-axis : feature  y-axis : allele
FeatRanks_heatmap


# In[ ]:


FreqComp = pd.DataFrame(0, index = range(NumAllele), columns = range(LenPep))

for i in range(LenPep):
    for j in range(LenRep) :
        FreqComp[i] = FreqComp[i] + FeatRanks_heatmap[LenPep*j+i]

FreqComp


# In[ ]:


clustermap(FreqComp, FeatRanks, LenPep, 'Alleles', 'Frequency Component')


# In[ ]:


NumRepComp = pd.DataFrame(0, index = range(NumAllele), columns = range(LenRep))

for i in range(LenRep):
    for j in range(LenPep) :
        NumRepComp[i] = NumRepComp[i] + FeatRanks_heatmap[LenPep*i+j]

NumRepComp


# In[ ]:


clustermap(NumRepComp, FeatRanks, LenRep, 'Alleles', 'Numeric Representation Component')


# In[ ]:


Total_FreqComp_Sum = []
for j in range(LenPep):
    Total_FreqComp_Sum.append(FreqComp[j].sum()/LenRep)

Total_NumRepComp_Sum = []
for i in range(LenRep):
    Total_NumRepComp_Sum.append(NumRepComp[i].sum()/LenPep)

Total_Features_Sum = []
for k in range(LenFeats):
    Total_Features_Sum.append(FeatRanks_heatmap[k].sum())


# In[ ]:


plt.plot(range(LenRep), Total_NumRepComp_Sum)
print(sum(Total_NumRepComp_Sum)*LenPep)


# In[ ]:


plt.plot(range(LenPep), Total_FreqComp_Sum)
print(sum(Total_FreqComp_Sum)*LenRep)


# In[ ]:


plt.plot(range(LenFeats), Total_Features_Sum)
print(sum(Total_Features_Sum))


# In[ ]:


FeatSetAccu


# In[ ]:


NumFeatUsed


# In[ ]:


NAllele = [] # Allele Frequency
NFeatUniq = NumFeatUsed['Number of features used'].unique().tolist()
NFeatUniq_S = sorted(NFeatUniq)  # sort() returns None, sorted() returns a sorted list


for NFeat in NFeatUniq_S :
    NAllele.append((NumFeatUsed['Number of features used'].loc[NumFeatUsed['Number of features used']==NFeat].count())*100/44)

bp = sns.barplot(x = NFeatUniq_S, y = NAllele)  # NFeat vs Percentage of NALlele
bp.set_yticks(range(0,50,5))
plt.show()


# In[ ]:


AS_cgaccu


# In[ ]:


ValAccu_BlindAccu


# In[ ]:


#PredAccu_AVG 
Sample_Size = []
#ValAccu_BlindAccu['Sample Size'] = np.zeros(len(ValAccu_BlindAccu))
for Alleles in Allele_list :
    Sample_Size.append(len(AS_tIS[Alleles]))
ValAccu_BlindAccu['Sample Size'] = Sample_Size

PASS = []
for i in range(len(ValAccu_BlindAccu)):
    PASS.append((ValAccu_BlindAccu.iloc[i,2])*(ValAccu_BlindAccu.iloc[i,3]))
ValAccu_BlindAccu['PA X SS'] = PASS

OverallPredictionAccuracy = sum(PASS)/sum(Sample_Size)
print('Overall Blind Test Accuracy : {}'.format(OverallPredictionAccuracy))


plt.figure(figsize=(15,8))
AllAccu = sns.barplot(x = ValAccu_BlindAccu['Allele'], y = ValAccu_BlindAccu['Prediction Accuracy'])  # NFeat vs Percentage of NALlele

AllAccu.set_xticklabels(AllAccu.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
plt.show()


# In[ ]:


FeatSetAccu
for i in range(NumAllele) :
    plt.plot(numFeat, FeatSetAccu.iloc[i,1:int(LenFeats/intv)+1])


# In[ ]:


# for ROC
#clf = load('SVMHLA_DPA1_01_03_DPB1_02_01.joblib')
#clf.score(X_train_all_sf, y_train_all_sf)


# In[ ]:


DataStats


# In[ ]:


DataStats['P_B'] 


# In[ ]:


ValAccu_BlindAccu['P_NB'] = DataStats['P_NB'] 
ValAccu_BlindAccu


# In[ ]:



sns.relplot(data=ValAccu_BlindAccu, x="Sample Size", y="Prediction Accuracy", hue ='P_NB', height = 5, aspect = 10/5)


# In[ ]:



zoomin = sns.relplot(data=ValAccu_BlindAccu, x="Sample Size", y="Prediction Accuracy", hue ='P_NB', height = 5, aspect = 10/5)

zoomin.ax.set_xlim(0,1200)
plt.show()


# In[ ]:



sns.relplot(data=ValAccu_BlindAccu, x="Sample Size", y="Validation Accuracy", hue ='P_NB',height = 5, aspect = 10/5)


# In[ ]:


ValAccu_BlindAccu.sort_values(by = 'Sample Size', ascending = False)


# In[ ]:





# In[ ]:




