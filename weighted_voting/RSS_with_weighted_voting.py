import numpy as np
import random
from scipy.stats import rankdata
from consensus import consensus_voting
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def consensus_RSS_Ozturk_N01_new2(H, K, N, theta, rho, c = 1, TIES = False):
    X_mean = theta
    sigma_sq_X = 1
    sigma_X = np.sqrt(sigma_sq_X)
    tau_sq_eps = sigma_sq_X*(1/rho**2 - 1)
    tau_eps = np.sqrt(tau_sq_eps)
    Lista_X = []
    Lista_ranks = []
    samples_ties = []
    #samples_weights = []
    #samples_ties1 = []
    #samples_ties2 = []
    samples_RSS = []
    random_user = np.random.randint(K, size=1)
    for n in range(N):
        sample_ties = []
        #sample_weights = []
        #sample_ties1 = []
        #sample_ties2 = []
        for sam in range(H):
            X = np.random.normal(X_mean,sigma_X,H)
            Lista_X.append(X)
            epsilons = []
            for k in range(K):
                epsilons.append(np.random.normal(0,tau_eps,H))
            Y = X + epsilons
            if TIES == True:
                Y_gwiazdka = Y/(c*sigma_X)
                Y_gw = np.round(Y_gwiazdka)
            else:
                Y_gw = Y
            Y_gw[np.where(Y_gw == -0.)] = 0.
            Y_gw_ = []
            for el in Y_gw:
                v = rankdata(el, method='ordinal') - 1
                Y_gw_.append(v)
        #takie same elementy
            Y_gw__ = []
            for el in Y_gw:
                records_array = el
                vals, inverse, count = np.unique(records_array, return_inverse=True,
                                return_counts=True)

                idx_vals_repeated = np.where(count > 1)[0]
                vals_repeated = vals[idx_vals_repeated]
                rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
                _, inverse_rows = np.unique(rows, return_index=True)
                res = np.split(cols, inverse_rows[1:])
                Y_gw__.append(res)
            lista1 = Y_gw__
            b = [[[Y_gw_[j][i] for i in el] for el in list(Y_gw__[j])]for j in range(len(Y_gw_))]
            lista2 = b
            for el in range(len(Y_gw_)):
                Y_gw_[el] = list(Y_gw_[el])
        #rangi z ties
            new_list_of_lists = [list(Y_gw_[el]) for el in range(len(Y_gw_))]
            for i in range(len(Y_gw__)):
                for f in range(len(lista2[i])):
                    for elm in list(lista1[i][f]):
                        new_list_of_lists[i][elm] = lista2[i][f]
            Lista_ranks.append(new_list_of_lists)
            usr_sampl_scl = np.zeros((H,K,H))
            for k in range(K):
                for h in range(H):
                    usr_sampl_scl[new_list_of_lists[k][h], k, h] = 1
            samples_RSS.append(X[np.argmax(usr_sampl_scl[sam, random_user, :])])

            result = consensus_voting(usr_sampl_scl)
            samples_logits = result.get('samples_logits')
            where_1 = np.where(samples_logits[:,sam] == max(samples_logits[:,sam]))
            if len(where_1[0]) > 1:
                #miara koncentracji tau
                new1 = np.zeros_like(samples_logits)
                for r in range(H):
                    new1[:,r] = samples_logits[:, r]*(r-sam)**2
                index = np.where(np.sum(new1, axis=1) == np.min(np.sum(new1, axis=1)[where_1]))[0]
                if len(index) > 1:
                    random_choice = np.random.choice(index,1)
                    sample_ties.append(X[random_choice])
                    #odds = np.exp(samples_logits[random_choice])
                    #sample_weights.append(odds / (1 + odds))

                else:
                    sample_ties.append(X[index])  
                    #odds = np.exp(samples_logits[index])
                    #sample_weights.append(odds / (1 + odds))
            elif len(where_1[0]) == 1:
                sample_ties.append(X[where_1])
                #odds = np.exp(samples_logits[where_1])
                #sample_weights.append(odds / (1 + odds))
            """
            user_weightet_cl = usr_sampl_scl.copy()
            indexes_right = np.where(user_weightet_cl == 1.)
            indexesR_ = np.array(indexes_right)
            indexesR_[0] = indexesR_[0]+1
            x = indexesR_[:,np.where(indexesR_[0] != indexesR_[0].max())[0]]
            y = tuple(tuple(sub) for sub in x)
            user_weightet_cl[y] = 0.5
            indexesL_ = np.array(indexes_right)
            indexesL_[0] = indexesL_[0]-1
            z = indexesL_[:,np.where(indexesL_[0] != -1)[0]]
            f = tuple(tuple(sub) for sub in z)
            user_weightet_cl[f] = 0.5

            result1 = consensus_voting(user_weightet_cl)
            samples_logits1 = result1.get('samples_logits')
            where_11 = np.where(samples_logits1[:,sam] == max(samples_logits1[:,sam]))
            if len(where_11[0]) > 1:
                #miara koncentracji tau
                new = np.zeros_like(samples_logits1)
                for r in range(H):
                    new[:,r] = samples_logits1[:, r]*(r-sam)**2
                index = np.where(np.sum(new, axis=1) == np.min(np.sum(new, axis=1)[where_11]))[0]
                if len(index) > 1:
                    random_choice1 = np.random.choice(index,1)
                    sample_ties1.append(X[random_choice1])
                else:
                    sample_ties1.append(X[index])  
            elif len(where_11[0]) == 1:
                sample_ties1.append(X[where_11])

            user_weightet_cl_norm = usr_sampl_scl.copy()
            new_matrix = np.abs(user_weightet_cl_norm - 1)
            where_zero = np.array(np.where(new_matrix == 0.))
            where_one = np.array(np.where(new_matrix == 1.))
            for i in range(where_zero.shape[1]):
                for j in range(where_one.shape[1]):
                    if np.all(where_zero[1:3][:,i] == where_one[1:3][:,j]):
                        y = tuple(where_one[:,j])
                        new_matrix[y] = np.abs(where_zero[0,i] - where_one[0,j])
                    else: 
                        continue
            usr_sampl_cl_norm = stats.norm.pdf(new_matrix, 0, 1)

            result2 = consensus_voting(usr_sampl_cl_norm)
            samples_logits2 = result2.get('samples_logits')
            where_12 = np.where(samples_logits2[:,sam] == max(samples_logits2[:,sam]))
            if len(where_12[0]) > 1:
                #miara koncentracji tau
                new12 = np.zeros_like(samples_logits2)
                for r in range(H):
                    new12[:,r] = samples_logits2[:, r]*(r-sam)**2
                index = np.where(np.sum(new12, axis=1) == np.min(np.sum(new12, axis=1)[where_12]))[0]
                if len(index) > 1:
                    random_choice2 = np.random.choice(index,1)
                    sample_ties2.append(X[random_choice2])
                else:
                    sample_ties2.append(X[index])  
            elif len(where_12[0]) == 1:
                sample_ties2.append(X[where_12])
            """
        #samples_weights.append(sample_weights)
        samples_ties.append(sample_ties)
        #samples_ties1.append(sample_ties1)
        #samples_ties2.append(sample_ties2)

    #OZTURK
    lista_ranks = Lista_ranks
    New_data_v = []
    for lists in lista_ranks:
        New_data = []   
        for search in range(H):
            j=0
            new_data = np.zeros((len(lists),H))
            for data in lists:
                i=0
                for sublist in data:
                    if type(sublist) == list:
                        for el in sublist:
                            if el == search:
                                new_data[j][i] = 1/len(sublist)
                                break
                            else:
                                new_data[j][i] = 0
                    else:
                        if sublist == search:
                            new_data[j][i] = 1
                            break
                        else:
                            new_data[j][i] = 0
                    i+=1
                j+=1
            New_data.append(np.matrix(new_data))
        v = [[New_data[i][el] for i in range(H)] for el in range(K)]
        New_data_v.append(v)
    samples_matrices = [[np.transpose(New_data_v[i][j]) for j in range(K)] for i in range(H*N)]
    samples_matrices_cycles = split_list(samples_matrices, wanted_parts=N)
    new_matrix_ = []
    for n in range(N):
        for h in range(H):
            k = 1
            old_matrix = samples_matrices_cycles[n][h][0]
            while k< K:
                old_matrix =  old_matrix + samples_matrices_cycles[n][h][k]
                k+=1
            new_matrix_.append(old_matrix/K)
    new_matrix = split_list(new_matrix_, wanted_parts=N)
#c = [[i, j, new_matrix[i][j][j].argmax()] for i in range(N) for j in range(H)]
    b = [np.transpose(el) for el in new_matrix_]
    bb = split_list(b, wanted_parts=N)
    oho = [[i, j, np.where(new_matrix[i][j][j] == np.max(new_matrix[i][j][j]))[1]] for i in range(N) for j in range(H)]
#tau_s_r_r
    for el in oho:
        if len(el[2]) > 1:
            wynik = [np.sum([((t-el[1])**2)*bb[el[0]][el[1]][f][0][t] for t in range(H)]) for f in el[2]]
            gdzie = np.where(wynik == np.min(wynik))
            if len(gdzie[0]) > 1:
                el[2] = random.choice(el[2][gdzie])
            else:
                el[2] = el[2][np.where(wynik == np.min(wynik))]
    wagi_Ozturk = [bb[i][j][k] for i,j,k in oho]
    wagi_Ozturk2 = [el[0][0] if type(el[0][0]) is np.ndarray else el[0] for el in wagi_Ozturk]
    Lista_X_ = split_list(Lista_X, wanted_parts=N)
    sample_Ozturk = [Lista_X_[i][j][k] for i,j,k in oho]
    wynik = [el[0] if type(el) is np.ndarray else el for el in sample_Ozturk]
    #return {'consensus' : samples_ties, 'weights' : samples_weights,'Ozturk': sample_Ozturk, 'RSS' : samples_RSS, 'X_mean' : X_mean, "variance" : sigma_sq_X, "consensus_w_0.5" : samples_ties1, "consensus_w_norm": samples_ties2}
    return {'consensus' : samples_ties, 'Ozturk': wynik, 'wagi_Ozturk': wagi_Ozturk2, 'RSS' : samples_RSS, 'X_mean' : X_mean, "variance" : sigma_sq_X}