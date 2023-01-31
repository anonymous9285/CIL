'''
utis.py
'''

import numpy as np
import torch
import os
#import tensorflow as tf
import pickle
import scipy
import pdb

'''
_get_session(model, sess = None)
disturb_graph_acnode(A, m)
disturb_graph_edges(A, m, k, indices_node = None)
disturb_graph_node_add_del(A, X, m, k, signal_type = 'label')
disturb_graph_node_label(X, m, k, signal_type ='label',  indices_node = None)
get_path(dir_name, folder)
pca_te(x, w, x_mean = None, x_norm = None)
pca_tr(x, energy = 0.9, is_mean = True, is_norm = True)
save_variable_list(x,filepath,is_shared = 0)
load_variable_list(filepath)
'''

def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A,BT)
    SqA = A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED    

def normalize_adj(A, type="AD"):
    if type == "DAD":
        # d is  Degree of nodes A=A+I
        # L = D^-1/2 A D^-1/2
        A = A + np.eye(A.shape[0])  # A=A+I
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)
        G = torch.from_numpy(G)
    elif type == "AD":
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)
    else:
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        D = np.diag(D)
        G = D - A
    return G

def get_knn(rois):
    n = rois.shape[0]
    knn = np.zeros((n, n))
    bboxes = rois[..., 1:5].cpu().numpy()
    # print(bboxes)
    # print(np.array(bboxes))
    ctrx = bboxes[..., 0]
    ctry = bboxes[..., 1]
    
    ctrxy = np.concatenate((ctrx.reshape(-1, 1), ctry.reshape(-1, 1)), axis=1)
    
    similarity_e = EuclideanDistances(ctrxy, ctrxy)
    
    knn_graph = np.argsort(similarity_e, axis=1)[:, :]
    for node in range(n):
        neighbors = knn_graph[node, 1:4] # 1:4(k=3)
        for i in neighbors.A[0]:
            knn[i, node] = 1
            knn[node, i] = 1
    return knn


def disturb_graph_acnode(A, m):
    h = np.where(np.sum(A[0:m,0:m], axis=1)==0)
    w = np.random.randint(m, size = len(h))
    A[h,w] = 1
    A[w,h] = 1
    return 


def disturb_graph_edges(A, m, k, indices_node = None):
    # A: n*n; dense/sparse
    # indices_node: only for adding edges
    n = A.shape[0]
    if k > 0: # add edges
        if indices_node is not None:
            h = np.random.choice(indices_node, size = k)
        else:
            h = np.random.randint(m, size = k)
        w = np.random.randint(m, size = k)
        A[h,w] = 1
        A[w,h] = 1
    elif k < 0: # del edges
        if scipy.sparse.issparse(A):
            A_coo = A
        else:
            A_coo = scipy.sparse.coo_matrix(A)
        nedge = A_coo.nnz
        k = np.min((nedge, -k))

        ix = np.random.randint(nedge, size=k)
        h, w = A_coo.row[ix], A_coo.col[ix]
        A[h,w] = 0
        A[w,h] = 0
    #disturb_graph_acnode(A, m)

    return #A


def disturb_graph_node_label(X, m, k, signal_type ='label',  indices_node = None):
    # X: [n, d]
    # m: num of nodes, n >= m
    if k == 0:
       return
    n, d = X.shape
    assert(k <= m)
    if indices_node is not None:
        k = len(indices_node)
    else:
        indices_node = np.random.randint(m, size = k)

    X[indices_node,:] = 0
    if signal_type == 'label':
        l = np.random.randint(d, size=k)
        X[indices_node, l] = 1
    else:
        assert(1==2)

    return #X
        
         
def disturb_graph_node_add_del(A, X, m, k, signal_type = 'label'):
    # A: n*n, dense/sparse
    # A_coo_row, A_coo_col: indices of non-zeros, 1D array
    # nodelables: 1D array
    # m: num of true nodes, due to mask mtx

    n = A.shape[0]
    if k > 0: # add nodes
        k = np.min((k, n-m))
     
        indices_node = np.arange(m, m + k)
        m = m + k
        ## set attributes of new nodes
        disturb_graph_node_label(X, m, k, signal_type, indices_node)
 
        ## add edges
        k_edge = np.ceil(np.count_nonzero(A[0:m-k, 0:m-k])/(m-k)/2)
        disturb_graph_edges(A, m, int(k*k_edge), indices_node)
        
    elif k < 0: # del nodes and corresponding edges
        k = np.min((m, -k))
        idx = np.random.permutation(m)[k:] #.reshape((-1,1))
        m = m - k
        X[0:m,:] = X[idx,:]
        X[m:m+k,:] = 0

        idx = idx.reshape((-1,1))
        A[0:m,0:m] = A[idx,idx.T]
        A[m:m+k, :] = 0
        A[:,m:m+k]  = 0
    return m#, A, X  # [A, X are updated]
        

def net_print(net, logger):
      ## print
    nlayer = len(net)
    for ii in range(nlayer):
        layer = net[ii]
        str1 = '{} layer:\n\t'.format(ii)
        for keyname in layer.keys():#sorted(layer.keys()):
            str1 = str1 + '{}:{}; '.format(keyname,layer[keyname])
        logger.info(str1)

def net_parsing(net, args, X_shape, n_class):

    n_concat, kk = -1, 0
    net[-1]['class_num'] = n_class
    flag_using_W = False
    for layer in net:
        if layer['name'] == 'input':

            layer['batch_size'] = np.max((args.batch_size,1))
            if args.subgraph_size < 10:
                layer['subgraph_size'] = X_shape[-2]
            else:
                layer['subgraph_size'] = args.subgraph_size
            layer['fea_dim'] = X_shape[-1]
            
            b_sz, sg_sz, k_adj = layer['batch_size'], layer['subgraph_size'], layer['k_adj']

            A_in_shape = np.int32([b_sz*sg_sz, k_adj])
            X_in_shape = np.int32([b_sz*sg_sz, layer['fea_dim']])
            A_out_shape, X_out_shape = A_in_shape, X_in_shape

            ## for random projection
            if layer.has_key('is_hash'):
                if layer['is_hash'] == True:
                    if layer.has_key('n_bucket'):
                        layer['n_bucket'] = np.max((layer['n_bucket'], 1))
                    else:
                        layer['n_bucket']= 1

                    if layer.has_key('n_hash'):
                        layer['n_hash']   = np.max((layer['n_hash'], 1))
                    else:
                        layer['n_hash'] = 1
                    assert(layer['n_bucket'] > 1)
            else:
                layer['is_hash'] = False
                

        elif layer['name'] == 'conv':
            assert(layer['ffun']=='max' or layer['ffun'] == 'sum' or layer['ffun'] == 'mean' or layer['ffun']=='hos')
            #assert(max_K >= layer['K'])
            X_out_shape[-1] = layer['F'] #layer['C']*layer['K']* (2*X_in_shape[-1])
            if layer['ffun'] == 'sum' or layer['ffun'] == 'mean':
                flag_using_W = True

        elif layer['name'] == 'pool':
            assert(layer['K'] > 0)
            if layer['ffun'] == 'sum' or layer['ffun'] == 'mean':
                flag_using_W = True

        elif layer['name'] == 'concat':
            n_concat = kk

        elif layer['name'] == 'feamapping':
            X_out_shape[-1] = layer['F']                
        
        elif layer['name'] == 'fc':
            A_out_shape = None
            X_out_shape = np.copy(X_in_shape)#[X_out_shape[0], layer['out_dim']]
            X_out_shape[-1] = layer['out_dim']
        
        elif layer['name'] == 'logres':
            A_out_shape = None
            X_out_shape = np.copy(X_in_shape)
            X_out_shape[-1] = layer['class_num']
            #X_out_shape = [X_out_shape[0], layer['class_num']]
        else:
            print('{} not exist'.format(layer['name']))
            assert(1==2)
        layer['A_in_shape'],layer['X_in_shape'] = np.copy(A_in_shape), np.copy(X_in_shape)
        layer['A_out_shape'],layer['X_out_shape'] = np.copy(A_out_shape), np.copy(X_out_shape)

        A_in_shape, X_in_shape = np.copy(A_out_shape), np.copy(X_out_shape)
        kk = kk + 1

    net[0]['is_using_W'] = flag_using_W
    net_print(net, args.logger)

    return net, n_concat

def net_parsing_dual(net1, net2, args, X1_shape, X2_shape, n_class):

    net1, k1 = net_parsing(net1, args, X1_shape, n_class)
    net2, k2 = net_parsing(net2, args, X2_shape, n_class)

    layer1, layer2 = net1[k1], net2[k2]
    assert(layer1['name']=='concat' and layer2['name']=='concat')

    shp1, shp2 = layer1['X_out_shape'],  layer2['X_out_shape']
    shp1[-1] = shp1[-1] + shp2[-1]
    layer1['X_out_shape'], layer2['X_out_shape'] = shp1, shp1

    layer1 = net1[k1+1]
    layer1['X_in_shape'] = shp1

    net2[k2+1:] = []

    ## print
    net_print(net1, args.logger)
    net_print(net2, args.logger)

    return net1, net2


def get_path(dir_name, folder):

    return os.path.join(dir_name, folder)


def pca_tr(x, energy = 0.9, is_mean = True, is_norm = True, is_white = False, rm_k_dim = 0):
    # x: n * d
    z = np.transpose(x) # d*n

    if is_mean:
        x_mean = np.mean(z,axis=1, keepdims = True)
        x_mean = np.asarray(x_mean, dtype = x.dtype)
        z = z-x_mean
        x_mean = np.transpose(x_mean)
    else:
        x_mean = 0

    if is_norm:
        x_norm = np.linalg.norm(z, axis = 1, keepdims=True)
        x_norm = np.asarray(x_norm, dtype = x.dtype)
        idx = np.where(x_norm < 1.0e-6)
        x_norm[idx] = 1.
        z = z/x_norm  
        x_norm = np.transpose(x_norm)
    else:
        x_norm = 1

    #
    d, n = z.shape
    #pdb.set_trace()
    if d >= n:
        ztz = np.dot(np.transpose(z), z) 
        a, v = np.linalg.eig(ztz)
        ind = np.argsort(a)[::-1] 
        ev = a[ind]
        v = v[:,ind]
       
        if energy <=1:
            r = np.cumsum(ev)/np.sum(ev)
            ind = np.where(r >= energy)
            dim = ind[0]
        else:
            dim = energy
        dim = dim + rm_k_dim

        a = ev[0:dim]
        a = np.diag(1/np.sqrt(a))
        v = v[:,0:dim]
        w = np.dot(np.dot(z,v),a) 
    else:
        zzt = np.dot(z, np.transpose(z))
        a, v = np.linalg.eig(zzt)
        ind = np.argsort(a)[::-1]
        ev = a[ind]
        v  = v[:,ind]

        if energy <=1:
            r = np.cumsum(ev)/np.sum(ev)
            ind = np.where(r >= energy)
            dim = ind[0]
        else:
            dim = energy
        dim = dim + rm_k_dim

        w = v[:,0:dim]

    ##
    w = np.asarray(w, dtype = x.dtype)

    #pdb.set_trace()
    if is_white:
        w = w/np.reshape(np.real(np.sqrt(ev[0:dim]))+1.0e-8, (1,dim))

    w = w[:, rm_k_dim:dim]
    dim = dim - rm_k_dim

    return (is_mean, x_mean, is_norm,  x_norm, w, ev, dim)

def pca_te(x, proj):
    # x: n*d
    # x_mean: 1*d
    # x_norm: 1*d
    # w: d*d2
    # output: n*d2
    is_mean, x_mean, is_norm, x_norm, w, _, _ = proj 
    if is_mean:
        z = x-x_mean
    else:
        z = x

    if is_norm:
        z = z/x_norm

    return np.dot(z, w)



def save_variable_list(x,filepath,is_shared = 0):
    if is_shared==1:
        y = []
        for xi in x:
            y.append(xi.get_value())
        fp = open(filepath,"wb")
        cPickle.dump(y,fp,protocol=-1)
        fp.close()
    else:
        fp = open(filepath,"wb")
        pickle.dump(x,fp,protocol=-1)
        fp.close()


def load_variable_list(filepath):
    fp = open(filepath,"rb")
    y=pickle.load(fp)
    fp.close()
    return y

