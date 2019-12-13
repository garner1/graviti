#!/usr/bin/env python
# coding: utf-8

# In[8]:


from scipy import ndimage, sparse
from scipy.linalg import eigh, inv, logm, norm
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

from IPython.display import clear_output


# In[9]:


def edges_rescaling(edges,scale): # edges are mat.data where mat is a sparse scipy matrix
    edges = np.log10(edges) # log rescale weights because they vary over many decades
    edges -= min(edges) # make them positive 
    edges /= max(edges)*1.0/scale # rescale from 0 to scale
    return edges

def build_omegaij(Kdata,Krow,Kcol,m):
    omegaIJ_data = np.zeros(Kdata.shape)
    omegaIJ_data = np.asfarray([Kdata[ind]*(1.0/m[Krow[ind]] + 1.0/m[Kcol[ind]]) for ind in range(omegaIJ_data.shape[0])])
    omegaIJ = sparse.coo_matrix((omegaIJ_data, (Krow, Kcol)), shape=(Krow.max()+1,Kcol.max()+1))
    return omegaIJ
def build_omegai(K,m):
    #0.5 to avoid double counting
    omegaI = 0.5*np.divide(K.sum(axis=1),m.reshape((m.shape[0],1)))
    return omegaI

def remove_col(mat,index_to_drop): #csr
    to_keep = list(set(range(mat.shape[1]))-set(index_to_drop))    
    mat = mat[:,to_keep]
    return mat
def remove_row(mat,index_to_drop): #csc
    to_keep = list(set(range(mat.shape[0]))-set(index_to_drop))    
    mat = mat[to_keep,:]
    return mat
def remove_2nodes(mat,nodes):
    mat = mat.tocoo()
    todrop1 = np.logical_or((mat.row==nodes[0]),(mat.row==nodes[1])).nonzero()[0]
    todrop2 = np.logical_or((mat.col==nodes[0]),(mat.col==nodes[1])).nonzero()[0]
    todrop = list(set(np.concatenate((todrop1,todrop2))))
    newdata=np.delete(mat.data,todrop)
    newrow=np.delete(mat.row,todrop)
    newcol=np.delete(mat.col,todrop)
    return sparse.coo_matrix((newdata, (newrow, newcol)), shape=mat.shape)
def remove_1node(mat,node):
    mat = mat.tocoo()
    todrop = np.logical_or((mat.row==node[0]),(mat.col==node[0])).nonzero()[0]
    todrop = list(set(todrop))
    newdata=np.delete(mat.data,todrop)
    newrow=np.delete(mat.row,todrop)
    newcol=np.delete(mat.col,todrop)
    return sparse.coo_matrix((newdata, (newrow, newcol)), shape=mat.shape)
def expand(Kdata,Krow,Kcol,omegaIJdata,omegaIJrow,omegaIJcol,idxs,m,g):
    for idx in idxs:
        newdata=K.data[idx]
        j=K.col[idx]
        Kdata,Krow,Kcol,omegaIJdata,omegaIJrow,omegaIJcol = expand1(Kdata,Krow,Kcol,omegaIJdata,omegaIJrow,omegaIJcol,newdata,m,g,j)
    return Kdata,Krow,Kcol,omegaIJdata,omegaIJrow,omegaIJcol
def expand1(Kdata,Krow,Kcol,omegaIJdata,omegaIJrow,omegaIJcol,newk,m,i,j):
    Kdata = np.append(Kdata,newk)
    Krow = np.append(Krow,i)
    Kcol = np.append(Kcol,j)
    omegaIJdata = np.append(omegaIJdata,newk*(1.0/m[i]+1.0/m[j]))
    omegaIJrow = np.append(omegaIJrow,i)
    omegaIJcol = np.append(omegaIJcol,j)
    #add symmetric
    Kdata = np.append(Kdata,newk)
    Krow = np.append(Krow,j)
    Kcol = np.append(Kcol,i)
    omegaIJdata = np.append(omegaIJdata,newk*(1.0/m[i]+1.0/m[j]))
    omegaIJrow = np.append(omegaIJrow,j)
    omegaIJcol = np.append(omegaIJcol,i)
    return Kdata,Krow,Kcol,omegaIJdata,omegaIJrow,omegaIJcol
def delete_nodes(Kdata,Krow,Kcol,omegaIJdata,omegaIJrow,omegaIJcol,idxs): #this is not symm wrt to (i,j)
    Kdata = np.delete(Kdata,idxs)
    Krow = np.delete(Krow,idxs)
    Kcol = np.delete(Kcol,idxs)
    omegaIJdata = np.delete(omegaIJdata,idxs)
    omegaIJrow = np.delete(omegaIJrow,idxs)
    omegaIJcol = np.delete(omegaIJcol,idxs)
    return Kdata,Krow,Kcol,omegaIJdata,omegaIJrow,omegaIJcol


# In[90]:


filename='/home/garner1/Work/pipelines/tissue2graph/npz/mat_XY_10nn.npz'
data = np.load('/home/garner1/Work/pipelines/tissue2graph/npz/X-XY_data.npz',allow_pickle=True)

K0 = sparse.load_npz(filename) #minus

# N = 1000000
# K = K0.copy()[:N,:N].tocoo()  #make a copy of the initial data
# m = np.ones(K0.shape[0])[:N] #initial masses
# pos = data['XY'][:N,:]

K = K0.copy().tocoo()  #make a copy of the initial data
m = np.ones(K0.shape[0]) #initial masses
pos = data['XY']

#initialize omegaIJ/I
omegaIJ = build_omegaij(K.data,K.row,K.col,m)
omegaI = build_omegai(K,m) 

# delete very small entries  
# threshold = np.finfo(np.float).eps
threshold = 9.9e-1

idxs_todelete = np.where(np.abs(K.data) < threshold)[0]
print(K.data.shape)
K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,idxs_todelete)
print(K.data.shape)


# In[91]:


import networkx as nx
import seaborn as sns; sns.set()
sns.set(style='white', rc={'figure.figsize':(50,50)})

init_mat = sparse.coo_matrix((K.data, (K.row, K.col)), shape=(K.row.max()+1, K.col.max()+1)) 

G = nx.from_scipy_sparse_matrix(init_mat) # if sparse matrix
eset = [(u, v) for (u, v, d) in G.edges(data=True)]
weights = [d['weight'] for (u, v, d) in G.edges(data=True)]

nx.draw_networkx_nodes(G, pos,alpha=0.0)
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')
plt.axis('off')

plt.savefig('./before_RG_plus.png',bbox_inches='tight')
plt.close()

# sns.set(style='white', rc={'figure.figsize':(10,10)})
# sns.distplot(weights)


# In[94]:


'''RG flow'''
condition = True
Imax0 = 0
IJmax0 = 0
counter = 0
while condition:
# for count in range(100):
    counter += 1
    #Find max btw node and edges
    IJmax_idx = np.where( omegaIJ.data==np.max(omegaIJ.data[np.nonzero(omegaIJ.data)]) )[0][0]
    i0 = np.where( omegaI==np.max(omegaI[np.nonzero(omegaI)]) )[0][0]
    Imax = omegaI[i0][0,0]
    IJmax = omegaIJ.data[IJmax_idx]
    maxtype = np.argmax([Imax,IJmax])
#     condition = (abs(Imax-Imax0) > 1e-16) or (abs(IJmax-IJmax0) > 1e-16)
    condition = np.count_nonzero(omegaI) > 900000
    Imax0 = Imax; IJmax0 = IJmax
#     clear_output()
    if Imax<=IJmax: #if edge
        if (counter == 1000):
            print('edge removal')
            print(counter,[Imax,IJmax],np.count_nonzero(omegaI),omegaIJ.data.shape[0])
            counter = 0
        i0 = K.row[IJmax_idx];j0 = K.col[IJmax_idx] #find max edge (i0,j0)  
        m = np.append(m,m[i0]+m[j0]) #add a center of mass node
        g = K.row.max()+1 # label the i0-j0 center of mass node
        pos = np.vstack((pos,[0.5*(pos[i0,0]+pos[j0,0]),0.5*(pos[i0,1]+pos[j0,1])]))
        idx_i0isrow = np.argwhere(K.row==i0) # idxs of (i0,j)
        idx_i0iscol = np.argwhere(K.col==i0) # idxs of (j,i0)
        idx_j0isrow = np.argwhere(K.row==j0) # idxs of (j0,i)
        idx_j0iscol = np.argwhere(K.col==j0) # idxs of (i,j0)
        js = np.setdiff1d([K.col[idx] for idx in np.union1d(idx_i0isrow,idx_j0isrow)],[i0,j0]) #nodes neighbours of i0 and j0
        for j in  js:
            idx_i0j = np.intersect1d(np.argwhere(K.row==i0),np.argwhere(K.col==j))
            idx_j0j = np.intersect1d(np.argwhere(K.row==j0),np.argwhere(K.col==j))
            newk = np.sum(np.append(K.data[idx_i0j],K.data[idx_j0j]))
            K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = expand1(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,newk,m,g,j)
            #remove i0 and j0 from K, omegaIJ
            K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.intersect1d(np.argwhere(K.row==i0),np.argwhere(K.col==j)))
            K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.intersect1d(np.argwhere(K.row==j),np.argwhere(K.col==i0)))
            K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.intersect1d(np.argwhere(K.row==j0),np.argwhere(K.col==j)))
            K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.intersect1d(np.argwhere(K.row==j),np.argwhere(K.col==j0)))
            #remove (i0,j0) from K, omegaIJ
            K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.intersect1d(np.argwhere(K.row==i0),np.argwhere(K.col==j0)))
            K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.intersect1d(np.argwhere(K.row==j0),np.argwhere(K.col==i0)))
        #update omegaI
        omegaI_g = np.array(sum([K.data[idx] for idx in np.argwhere(K.row==g)])*1.0/m[g]).reshape(1,1)
        omegaI = np.append(omegaI,omegaI_g,0)
        for j in js:
            omegaI[j] = sum([K.data[idx] for idx in np.argwhere(K.row==j)])*1.0/m[j]
        omegaI[i0] = 0.0; omegaI[j0] = 0.0
        
    if Imax>IJmax: #if node
        if (counter == 1000):
            print('node removal')
            print(counter,[Imax,IJmax],np.count_nonzero(omegaI),omegaIJ.data.shape[0])
            counter = 0
        idx_i0isrow = np.argwhere(K.row==i0) # idxs of (i0,j)
        idx_i0iscol = np.argwhere(K.col==i0) # idx of (i,i0)
        js = np.unique(K.col[idx_i0isrow]) # nn j in (i0,j)
        connectivity = omegaI[i0]*m[i0]
        for i in js:
            idx_ri = np.argwhere(K.row==i)
            idx_ci = np.argwhere(K.col==i)
            for j in js[np.argwhere(js==i)[0][0]+1:]:
                idx_cj = np.argwhere(K.col==j)
                idx_rj = np.argwhere(K.row==j)
                idx_ij = np.intersect1d(idx_ri,idx_cj)
                idx_ji = np.intersect1d(idx_rj,idx_ci)
                idx_ii0 = np.intersect1d(idx_ri,idx_i0iscol)    
                idx_i0j = np.intersect1d(idx_i0isrow,idx_cj)
                if idx_ij.shape[0]>0: #update edge value
                    K.data[idx_ij] = np.sum(np.append(K.data[idx_ij],K.data[idx_ii0]*K.data[idx_i0j]/connectivity))
                    K.data[idx_ji] = K.data[idx_ij]
                    omegaIJ.data[idx_ij] = K.data[idx_ij]*(1.0/m[i]+1.0/m[j])
                    omegaIJ.data[idx_ij] = omegaIJ.data[idx_ji]
                if idx_ij.shape[0]==0: #create a new edge
                    newk = K.data[idx_ii0]*K.data[idx_i0j]/connectivity
                    if newk >= threshold:
                        K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = expand1(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,newk,m,i,j)
        #remove i0 from K, omegaIJ
        K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.argwhere(K.row==i0))
        K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.argwhere(K.col==i0))
        #update omegaI
        for j in js:
            omegaI[j] = sum([K.data[idx] for idx in np.argwhere(K.row==j)])*1.0/m[j]
        #remove i0 omegaI
        omegaI[i0] = 0.0


# In[95]:


#minus here:
fixed_mat = sparse.coo_matrix((K.data, (K.row, K.col)), shape=(K.row.max()+1, K.col.max()+1))

import networkx as nx
import seaborn as sns; sns.set()
sns.set(style='white', rc={'figure.figsize':(50,50)})
G = nx.from_scipy_sparse_matrix(fixed_mat) # if sparse matrix
eset = [(u, v) for (u, v, d) in G.edges(data=True)]
weights = [d['weight'] for (u, v, d) in G.edges(data=True)]

nx.draw_networkx_nodes(G, pos,alpha=0.0)
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')
plt.axis('off')

plt.savefig('./after_RG_plus.png',bbox_inches='tight')
plt.close()


# In[ ]:




