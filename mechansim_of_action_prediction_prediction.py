__author__ = 'mohan'
import pandas as pd
from networkx.algorithms import bipartite
import networkx as nx
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_curve, auc
import time
from scipy import sparse
from gbssl import LGC,HMN,PARW,CAMLP,OMNI

np.random.seed(1)


def get_ground_truth():
    # load data
    df = pd.read_csv("Datasets/drug_gene_assosciation_data.txt",
                      sep = "\t",names = ["gene", "drugs","function"])

    src = df["function"]
    dest = df["gene"]
    #
    # src = df["location"]
    # dest = df["gene"]

    function_graph = nx.DiGraph()
    function_graph.add_nodes_from(src, bipartite=0)
    function_graph.add_nodes_from(dest, bipartite=1)
    edgelist = zip(src,dest)
    function_graph.add_edges_from(edgelist)
    function_nodes = set(n for n,d in function_graph.nodes(data=True) if d['bipartite']==0)
    gene_nodes = set(n for n,d in function_graph.nodes(data=True) if d['bipartite']==1)
    function_nodes = sorted(list(function_nodes))
    gene_nodes = sorted(list(gene_nodes))
    Ground_Truth_Matrix = nx.algorithms.bipartite.biadjacency_matrix(function_graph,
                                                                     row_order=function_nodes,
                                                                     column_order=gene_nodes,
                                                                     format= "lil")


    node_label_ids= []
    #label_ids =[]
    for u,v in function_graph.edges:
        node_label_ids.append((gene_nodes.index(v),function_nodes.index(u)))

    node_label_ids.sort(key=lambda tup: tup[0])
    GTL = []
    for u,v in node_label_ids:
        GTL.append(v)



    return Ground_Truth_Matrix,GTL,function_nodes,gene_nodes

def get_genetic_interaction_graph(gene_nodes):
    coexp_graph = nx.read_edgelist("Datasets/coexp_scores.txt", delimiter= " ",
                     nodetype=str,create_using=nx.Graph(),
                     data=(('weight',float),))
    cooccur_graph = nx.read_edgelist("Datasets/cooccurence_scores.txt", delimiter= " ",
                     nodetype=str,create_using=nx.Graph(),
                     data=(('weight',float),))

    experimental_graph = nx.read_edgelist("Datasets/experimental_scores.txt", delimiter= " ",
                     nodetype=str,create_using=nx.Graph(),
                     data=(('weight',float),))

    textmining_graph = nx.read_edgelist("Datasets/textmining_scores.txt", delimiter= " ",
                     nodetype=str,create_using=nx.Graph(),
                     data=(('weight',float),))


    all_graphs = [coexp_graph,cooccur_graph,experimental_graph,textmining_graph]
    F = nx.compose_all(all_graphs)
    X = nx.to_scipy_sparse_matrix(F, nodelist = gene_nodes)


    return X

def heat_diffusion_model(TG,t,idx_test,score_harmonic):
    # F = nx.compose(vp_graph,p_graph)
    # graph_matrix = nx.to_scipy_sparse_matrix(F, nodelist = total_people_nodes)
    graph_matrix = sparse.csgraph.laplacian(genetic_int_graph,normed=True)
    mak = TG.toarray().copy()
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
        state_matrix_new = normalize(state_matrix_new, axis = 0, norm = "l1")
        state_matrix = state_matrix_new.copy()

    return state_matrix

def get_best_time(Y,train_idx,score_harmonic):
    iteration = np.arange(0.05,5.05,0.05)
    #iteration = np.arange(0.1,1.1,0.1)
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
            GT =  Ground_Truth_Function_Drugs.toarray()

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
            #NL[mask_idx] = np.nan
            #############################################
            train_nodes = []
            labs = []
            for i in range(len(mask_idx[0])):
                train_nodes.append(mask_idx[1][i])
                labs.append(mask_idx[0][i])

            train_nodes = np.array(train_nodes)
            labs = np.array(labs)
            # F = nx.compose(vp_graph,p_graph)
            # X = nx.to_scipy_sparse_matrix(F, nodelist = total_people_nodes)
            lgc = LGC(graph=genetic_int_graph,alpha=alp)
            lgc.fit(np.array(train_nodes),np.array(labs))

            P =  lgc.predict_proba(np.arange(n))
            P= P.T
            GT =  Ground_Truth_Function_Drugs.toarray()

            prec, recall, _ = precision_recall_curve(GT[target_idx], P[target_idx])

            acc[cnt] = auc(recall, prec)
            cnt+=1
        cv_results[alp] = acc.mean()

    kL = sorted(cv_results, key=cv_results.get, reverse=True)
    #print sort_the_nodes
    print "Best alpha:", kL[0]
    return kL[0]
def get_boundary_score(train_nodes,labs,TG,train_idx):

    score_harmonic = get_harmonic_score(train_nodes,labs)
    t = get_best_time(TG,train_idx,score_harmonic)

    score_heat = heat_diffusion_model(TG,t,train_idx,score_harmonic)
    score = score_harmonic + score_heat
    #print score.A
    return score


def get_camlp_score(train_nodes,labs):

    camlp = CAMLP(graph=genetic_int_graph)
    camlp.fit(np.array(train_nodes),np.array(labs))

    P =  camlp.predict_proba(np.arange(n))
    return P.T

def get_lgc_score(train_nodes,labs,Y,train_idx):

    alp = get_best_alpha(Y,train_idx)
    lgc = LGC(graph=genetic_int_graph,alpha=alp)

    lgc.fit(np.array(train_nodes),np.array(labs))

    P =  lgc.predict_proba(np.arange(n))
    return P.T

def get_omni_score(train_nodes,labs):

    omni = OMNI(graph=genetic_int_graph)

    omni.fit(np.array(train_nodes),np.array(labs))

    P =  omni.predict_proba(np.arange(n))

    return P.T


def get_harmonic_score(train_nodes,labs):

    hmn = HMN(graph=genetic_int_graph)
    hmn.fit(np.array(train_nodes),np.array(labs))

    P =  hmn.predict_proba(np.arange(n))
    return P.T

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
    #P = get_harmonic_score(train_nodes,labs)
    #P = get_lgc_score(train_nodes,labs,Tc,orig_ids)
    #P = get_camlp_score(train_nodes,labs)
    #P = get_omni_score(train_nodes,labs)
    P = get_boundary_score(train_nodes,labs,Tc,orig_ids)

    GT =  Ground_Truth_Function_Drugs.toarray()

    prec, recall, _ = precision_recall_curve(GT[target_idx], P[target_idx])
    #print auc(recall, prec)
    return auc(recall, prec)

if __name__ == '__main__':

    Ground_Truth_Function_Drugs,GTL,func_nodes,gene_nodes = get_ground_truth()

    m = Ground_Truth_Function_Drugs.shape[0]
    n = Ground_Truth_Function_Drugs.shape[1]
    genetic_int_graph = get_genetic_interaction_graph(gene_nodes)

    print "Genes:",n
    print "Functions",m
    print "Number of Functions Gene Realtionship:",Ground_Truth_Function_Drugs.count_nonzero()

    SZ = m * n
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
        AUC_test[cnt] = innerfold(Ground_Truth_Function_Drugs, train_split, test_split)
        print "time to complete the folds:",time.time()-start,":seconds"

        cnt += 1

    print "All Testing Fold Results:",AUC_test
    print('AUC-PR Test Mean / Std: %f / %f' % (AUC_test.mean(), AUC_test.std()))