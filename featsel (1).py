import numpy as np
from svmtune import tuneSVM
from sklearn.svm import SVC
from kernelfunc import KN

# function featSelect
def featSelect(X_train_all, y_train_all, X_train_sets, y_train_sets, X_test_sets, y_test_sets, LenFeats, C_range, gamma_range, redF):
    allRank = np.zeros((len(X_train_sets), LenFeats ), dtype=float)
    
    # X_train_sets[1], X_train_sets[2], X_train_sets[3]...
    for i in range(0,len(X_train_sets)) :
        curFeats = []
        for j in range(0,LenFeats) :
            curFeats.append(j)
        
        remFeats = []
        
        while (len(curFeats)>2) :
            tunePer = tuneSVM(X_train_all, y_train_all, X_train_sets[:,:,curFeats], y_train_sets, X_test_sets[:,:,curFeats], y_test_sets, costs= C_range, gammas = gamma_range)
            
            max_accu_id = tunePer[:,2].argmax(axis = 0) 
            bp_c = tunePer[max_accu_id, 0] 
            bp_g = tunePer[max_accu_id, 1]
            ba = tunePer[max_accu_id, 2]
                                    
            svc = SVC(kernel = 'rbf', C=bp_c, gamma=bp_g).fit(np.transpose(X_train_sets[i,:,curFeats]), y_train_sets[i])
            
            alpha = np.zeros(len(X_train_sets[i])) 
            alpha[svc.support_] = abs(svc.dual_coef_) 
            
            labs = np.zeros(len(X_train_sets[i]))  
            labs[svc.support_] = np.sign(svc.dual_coef_)
            const = -0.5*(np.dot(alpha,alpha.transpose()))*(np.dot(labs,labs.transpose()))
            K = KN(np.transpose(X_train_sets[i,:,curFeats]), gamma = bp_g)
            
            crit = []
            for f in range (0,len(curFeats)) :
                Kp = KN(np.delete(np.transpose(X_train_sets[i,:,curFeats]), f, axis=1), gamma = bp_g)
                crit.append(np.sum(const*(K-Kp)))
            if redF ==0 :
                cT = np.max(crit)
            else :
                cT = np.flip(np.sort(crit))[np.ceil(len(curFeats)*redF).astype(int)]
            
            ri_crit = np.flip(np.argsort(crit))  # ri_crit : index of ranked crit in descending way
            ri_crit_cF = []
            for n in ri_crit :
                ri_crit_cF.append(curFeats[n])
            
            tocurFeats = []
            for k in ri_crit_cF :
                if crit[curFeats.index(k)] >= cT :
                    remFeats.append(k)
                else :
                    tocurFeats.append(k)
            curFeats = tocurFeats
                       
        remFeats = remFeats + curFeats
        allRank[i,:] = remFeats
    
    return allRank