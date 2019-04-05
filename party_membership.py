__author__ = 'mohan'
from rdflib import Graph
import networkx as nx
from networkx.algorithms import bipartite
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold
from scipy import sparse
from gbssl import LGC,HMN,PARW,CAMLP,OMNI

np.random.seed(20)

def get_ground_truth():
    g = Graph()
    g.parse("Datasets/us-presidents.rdf")

    all_nodes = set()
    all_rel = set()
    party_graph = nx.DiGraph()
    vice_president_graph = nx.DiGraph()
    president_graph = nx.DiGraph()


    for s,p,o in g:
        subj = s.rsplit('/', 1)[-1]
        pred = p.rsplit('/', 1)[-1]
        obj = o.rsplit('/', 1)[-1]

        if pred == "party":
            party_graph.add_node(obj, bipartite = 0)
            party_graph.add_node(subj,bipartite = 1)
            party_graph.add_edge(obj,subj) ##party person graph
            vice_president_graph.add_node(subj)
            president_graph.add_node(subj)

        if pred == "vicePresident":
            vice_president_graph.add_edge(subj,obj)
            party_graph.add_node(subj,bipartite = 1)
            party_graph.add_node(obj,bipartite = 1)
            president_graph.add_node(subj)
            president_graph.add_node(obj)

        if pred == "president":
            president_graph.add_edge(subj,obj)
            party_graph.add_node(subj,bipartite = 1)
            party_graph.add_node(obj,bipartite = 1)
            vice_president_graph.add_node(subj)
            vice_president_graph.add_node(obj)


        # print nx.info(party_graph)
        # print nx.info(president_graph)
        # print nx.info(vice_president_graph)
        all_nodes.add(subj)
        all_nodes.add(obj)
        all_rel.add(pred)



    total_party = set(n for n,d in party_graph.nodes(data=True) if d['bipartite']==0)
    total_people = set(n for n,d in party_graph.nodes(data=True) if d['bipartite']==1)
    total_people_nodes = sorted(list(total_people))
    total_party_nodes = sorted(list(total_party))

    print len(total_people_nodes)
    node_ids = []
    label_ids = []

    isl =  nx.isolates(party_graph)
    #print list(isl)

    Ground_Truth_Matrix = nx.algorithms.bipartite.biadjacency_matrix(party_graph,row_order=total_party_nodes,
                                                                         column_order=total_people_nodes)
    dict = {}
    for u,v in party_graph.edges:
        #print total_party_nodes.index(u),total_people_nodes.index(v)
        #print v,u
        dict[total_people_nodes.index(v)] = total_party_nodes.index(u)


    for i in isl:
       dict[total_people_nodes.index(i)] = -1



    return Ground_Truth_Matrix,dict.values(),vice_president_graph,president_graph,total_party_nodes,total_people_nodes

def get_best_alpha(Y,train_idx):
    iteration = np.arange(0.01,1.1,0.01)
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
            F = nx.compose(vp_graph,p_graph)
            X = nx.to_scipy_sparse_matrix(F, nodelist = total_people_nodes)
            lgc = LGC(graph=X,alpha=alp)
            lgc.fit(np.array(train_nodes),np.array(labs))

            P =  lgc.predict_proba(np.arange(n))
            P= P.T
            GT =  Ground_Truth_PM.toarray()

            prec, recall, _ = precision_recall_curve(GT[target_idx], P[target_idx])

            acc[cnt] = auc(recall, prec)
            cnt+=1
        cv_results[alp] = acc.mean()

    kL = sorted(cv_results, key=cv_results.get, reverse=True)
    #print sort_the_nodes
    print "Best alpha:", kL[0]
    return kL[0]


def heat_diffusion_model(TG,t,idx_test,score_harmonic):
    F = nx.compose(vp_graph,p_graph)
    graph_matrix = nx.to_scipy_sparse_matrix(F, nodelist = total_people_nodes)
    graph_matrix = sparse.csgraph.laplacian(graph_matrix,normed=True)
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
            GT =  Ground_Truth_PM.toarray()

            prec, recall, _ = precision_recall_curve(GT[target_idx], heat_score[target_idx])

            acc[cnt] = auc(recall, prec)
            cnt+=1
        cv_results[t] = acc.mean()

    kL = sorted(cv_results, key=cv_results.get, reverse=True)
    #print sort_the_nodes
    print "Best Time:", kL[0]
    return kL[0]



def get_boundary_score(train_nodes,labs,TG,train_idx):

    score_harmonic = get_harmonic_score(train_nodes,labs)
    t = get_best_time(TG,train_idx,score_harmonic)

    score_heat = heat_diffusion_model(TG,t,train_idx,score_harmonic)
    score = score_harmonic + score_heat
    #print score.A
    return score


def get_camlp_score(train_nodes,labs):
    GG = nx.compose(vp_graph,p_graph)
    X = nx.to_scipy_sparse_matrix(GG, nodelist = total_people_nodes)
    camlp = CAMLP(graph=X)
    camlp.fit(np.array(train_nodes),np.array(labs))

    P =  camlp.predict_proba(np.arange(n))
    return P.T

def get_harmonic_score(train_nodes,labs):
    GG = nx.compose(vp_graph,p_graph)
    X = nx.to_scipy_sparse_matrix(GG, nodelist = total_people_nodes)
    hmn = HMN(graph=X)
    hmn.fit(np.array(train_nodes),np.array(labs))

    P =  hmn.predict_proba(np.arange(n))
    return P.T

def get_lgc_score(train_nodes,labs,Y,train_idx):
    GG = nx.compose(vp_graph,p_graph)
    X = nx.to_scipy_sparse_matrix(GG, nodelist = total_people_nodes)
    alp = get_best_alpha(Y,train_idx)
    lgc = LGC(graph=X,alpha=alp)

    lgc.fit(np.array(train_nodes),np.array(labs))

    P =  lgc.predict_proba(np.arange(n))
    return P.T

def get_omni_score(train_nodes,labs):
    GG = nx.compose(vp_graph,p_graph)
    X = nx.to_scipy_sparse_matrix(GG, nodelist = total_people_nodes)
    omni = OMNI(graph=X)

    omni.fit(np.array(train_nodes),np.array(labs))

    P =  omni.predict_proba(np.arange(n))

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

    GT =  Ground_Truth_PM.toarray()


    prec, recall, _ = precision_recall_curve(GT[target_idx], P[target_idx])
    #print auc(recall, prec)
    return auc(recall, prec)

if __name__ == '__main__':
    Ground_Truth_PM,\
    GTL,\
    vp_graph,p_graph,\
    total_party_nodes,\
    total_people_nodes = get_ground_truth()

    m =Ground_Truth_PM.shape[0]
    n = Ground_Truth_PM.shape[1]
    SZ = m * n

    #np.random.shuffle(LM.toarray().T)


    # Do cross-validation
    FOLDS = 10
    AUC_train = np.zeros(FOLDS)
    AUC_test = np.zeros(FOLDS)
    IDX = list(range(SZ))
    kfold = KFold(FOLDS, True, 1)
    cnt = 0
    for train_split, test_split in kfold.split(IDX):
        print "Folds:",cnt
        AUC_test[cnt] = innerfold(Ground_Truth_PM, train_split, test_split)

        cnt += 1

    print "yo testing ko ho:",AUC_test
    print('AUC-PR Test Mean / Std: %f / %f' % (AUC_test.mean(), AUC_test.std()))








