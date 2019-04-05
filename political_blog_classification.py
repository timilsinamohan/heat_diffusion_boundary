__author__ = 'mohan'
from scipy.io import loadmat
import numpy as np
import networkx as nx
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_curve, auc
import time
from scipy import sparse
from gbssl import LGC,HMN,PARW,CAMLP,OMNI

np.random.seed(20)

def get_ground_truth():
    G = nx.read_gml('Datasets/polblogs.gml',label= 'id')
    true_labels = []

    for nodes,attr in G.nodes(data=True):
        if attr['value'] == 0:
            true_labels.append(0)
        else:
            true_labels.append(1)



    n_classes = len(np.unique(true_labels))
    n = len(true_labels)
    Labels = np.zeros((n,2),dtype=np.int)
    #print Labels


    Label_Matrix = np.zeros((n,n_classes),dtype=np.int)

    Labels[:,0]= np.arange(n)
    Labels[:,1]= true_labels[:]


    Label_Matrix[Labels[:,0],Labels[:,1]] = 1

    return Label_Matrix.T


def get_harmonic_score(train_nodes,labs):

    hmn = HMN(graph=graph_matx)
    hmn.fit(np.array(train_nodes),np.array(labs))

    P =  hmn.predict_proba(np.arange(n))
    return P.T


def heat_diffusion_model(TG,t,idx_test,score_harmonic):
    # F = nx.compose(vp_graph,p_graph)
    # graph_matrix = nx.to_scipy_sparse_matrix(F, nodelist = total_people_nodes)
    graph_matrix = sparse.csgraph.laplacian(graph_matx,normed=True)
    mak = TG.copy()
    mak = np.asarray(mak,np.float64)
    iteration = 30
    idx_test = np.unravel_index(idx_test, (m,n))
    #print "I am here:",mak

    mak[idx_test] = np.nan
    fzero = np.where(np.isnan(mak),
                     np.ma.array(mak, mask=np.isnan(mak)).mean(axis=1)[:, np.newaxis], mak)
    C = fzero - score_harmonic
    I = np.eye(n,n,dtype=np.float64)
    V = I - (t/iteration) * graph_matrix
    state_matrix = C.copy()

    for j in xrange(iteration):
        state_matrix_new = V.dot(state_matrix.T).T
        state_matrix_new = normalize(state_matrix_new, axis = 1, norm = "l1")
        state_matrix = state_matrix_new.copy()

    return state_matrix

def get_best_time(Y,train_idx,score_harmonic):
    iteration = np.arange(0.05,5.05,0.05)
    cv_results = {}
    for t in iteration:
        FOLDS = 5
        acc = np.zeros(FOLDS)
        IDX = train_idx[:]
        kfold = KFold(FOLDS, True, 1)
        cnt = 0
        for train_split, test_split in kfold.split(IDX):
            NL = Y.copy()
            mask_idx = np.unravel_index(test_split, (m, n))
            target_idx = np.unravel_index(test_split, (m, n))
            NL[mask_idx] = 0
            #NL[mask_idx] = np.nan
            heat_score = heat_diffusion_model(NL,t,test_split,score_harmonic)
            GT =  Ground_Truth_labels.copy()

            prec, recall, _ = precision_recall_curve(GT[target_idx], heat_score[target_idx])

            acc[cnt] = auc(recall, prec)
            cnt+=1
        cv_results[t] = acc.mean()

    kL = sorted(cv_results, key=cv_results.get, reverse=True)
    #print sort_the_nodes
    print "Best Time:", kL[0]
    return kL[0]

def get_best_alpha(Y,train_idx):
    iteration = np.arange(0.1,1.1,0.1)
    cv_results = {}
    for alp in iteration:
        FOLDS = 5
        acc = np.zeros(FOLDS)
        IDX = train_idx[:]
        kfold = KFold(FOLDS, True, 1)
        cnt = 0
        for train_split, test_split in kfold.split(IDX):
            NL = Y.copy()
            mask_idx = np.unravel_index(train_split, (m, n))
            target_idx = np.unravel_index(test_split, (m, n))
            NL[mask_idx] = 0

            train_nodes = []
            labs = []
            for i in range(len(mask_idx[0])):
                train_nodes.append(mask_idx[1][i])
                labs.append(mask_idx[0][i])

            train_nodes = np.array(train_nodes)
            labs = np.array(labs)

            lgc = LGC(graph=graph_matx,alpha=alp)
            lgc.fit(np.array(train_nodes),np.array(labs))

            P =  lgc.predict_proba(np.arange(n))
            P= P.T
            GT =  Ground_Truth_labels.copy()

            #print P[target_idx]

            prec, recall, _ = precision_recall_curve(GT[target_idx], P[target_idx])

            acc[cnt] = auc(recall, prec)
            cnt+=1
        cv_results[alp] = acc.mean()

    kL = sorted(cv_results, key=cv_results.get, reverse=True)

    print "Best alpha:", kL[0]
    return kL[0]


def get_boundary_score(train_nodes,labs,TG,train_idx):

    score_harmonic = get_harmonic_score(train_nodes,labs)
    t = get_best_time(TG,train_idx,score_harmonic)

    score_heat = heat_diffusion_model(TG,t,train_idx,score_harmonic)
    score = score_harmonic + score_heat

    return score


def get_camlp_score(train_nodes,labs):

    camlp = CAMLP(graph=graph_matx)
    camlp.fit(np.array(train_nodes),np.array(labs))

    P =  camlp.predict_proba(np.arange(n))
    return P.T

def get_lgc_score(train_nodes,labs,Y,train_idx):

    alp = get_best_alpha(Y,train_idx)
    lgc = LGC(graph=graph_matx,alpha=alp)

    lgc.fit(np.array(train_nodes),np.array(labs))

    P =  lgc.predict_proba(np.arange(n))
    return P.T

def get_omni_score(train_nodes,labs):

    omni = OMNI(graph=graph_matx)

    omni.fit(np.array(train_nodes),np.array(labs))

    P =  omni.predict_proba(np.arange(n))

    return P.T




def get_graph_matrix():

        # load data
    G = nx.read_gml('polblogs.gml',label= 'id')

    X = nx.to_scipy_sparse_matrix(G)

    return X


def innerfold(T, mask_idx, target_idx):
    Tc = T.copy()
    orig_ids = target_idx[:]
    target_idx = np.unravel_index(target_idx, (m,n))
    mask_idx = np.unravel_index(mask_idx, (m,n))
    # set values to be predicted to zero

    Tc[target_idx ] = 0
    train_nodes = []
    labs = []
    for i in range(len(mask_idx[0])):
        train_nodes.append(mask_idx[1][i])
        labs.append(mask_idx[0][i])

    train_nodes = np.array(train_nodes)
    labs = np.array(labs)
    #score = get_harmonic_score(train_nodes,labs)
    #score = get_lgc_score(train_nodes,labs,Tc,orig_ids)
    #score = get_camlp_score(train_nodes,labs)
    #score = get_omni_score(train_nodes,labs)
    score = get_boundary_score(train_nodes,labs,Tc,orig_ids)

    GT =  Ground_Truth_labels.copy()

    prec, recall, _ = precision_recall_curve(GT[target_idx], score[target_idx])
    #print auc(recall, prec)
    return auc(recall, prec)


if __name__ == '__main__':

    Ground_Truth_labels = get_ground_truth()
    m,n = Ground_Truth_labels.shape
    graph_matx = get_graph_matrix()

    print m,n

    SZ = n * m
    # Do cross-validation
    FOLDS = 10
    AUC_train = np.zeros(FOLDS)
    AUC_test = np.zeros(FOLDS)
    IDX = list(range(SZ))
    kfold = KFold(FOLDS, True, 1)
    cnt = 0
    for train_split, test_split in kfold.split(IDX):
        start = time.time()
        print "Folds:",cnt
        AUC_test[cnt] = innerfold(Ground_Truth_labels, test_split, test_split)
        print "Time to complete folds:",time.time()-start

        cnt += 1

    print "yo testing ko ho:",AUC_test
    print('AUC-PR Test Mean / Std: %f / %f' % (AUC_test.mean(), AUC_test.std()))
