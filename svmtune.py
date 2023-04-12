import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# function tuneSVM
def tuneSVM(X_train_all, y_train_all, X_train_sets, y_train_sets, X_test_sets, y_test_sets, costs, gammas):
    
    perf = []
    for c in costs :
        for g in gammas :
            auc = []
            cgauc = []
            for i in range(0,len(X_train_sets)) :
                svc = SVC(kernel = 'rbf', C =c, gamma=g).fit(X_train_sets[i], y_train_sets[i])
                auc.append(roc_auc_score(y_test_sets[i], svc.decision_function(X_test_sets[i])))
                
            cgauc = [c,g, np.mean(auc)]
            perf.append(cgauc)
    perf = np.array(perf)
    return perf
