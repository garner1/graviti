import warnings
warnings.filterwarnings('ignore')
import cdlib
from cdlib import algorithms
from cdlib.utils import convert_graph_formats
from cdlib import viz
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.measure import label, regionprops
from scipy.sparse import csr_matrix,lil_matrix,coo_matrix
from scipy.linalg import eigh, inv, logm, norm
from  scipy import ndimage,sparse
import cv2
import os
import sys
import csv
import glob
import umap
import cdlib
from cdlib import algorithms
from cdlib.utils import convert_graph_formats
from cdlib import viz
# import PyQt5
# PyQt5.QtWidgets.QApplication.setAttribute(PyQt5.QtCore.Qt.AA_EnableHighDpiScaling, True)
##################################################
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
    omegaI = 0.5*np.divide(K.sum(axis=1),m.reshape((m.shape[0],1)))     #0.5 to avoid double counting
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
    #add (i,j)
    Kdata = np.append(Kdata,newk)
    Krow = np.append(Krow,i)
    Kcol = np.append(Kcol,j)
    omegaIJdata = np.append(omegaIJdata,newk*(1.0/m[i]+1.0/m[j]))
    omegaIJrow = np.append(omegaIJrow,i)
    omegaIJcol = np.append(omegaIJcol,j)
    #add (j,i)
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

def covd(mat):
    ims = coo_matrix(mat)
    imd = np.pad( mat.astype(float), (1,1), 'constant')
    [x,y,I] = [ims.row,ims.col,ims.data]  

    Ix = [] #first derivative in x
    Iy = [] #first derivative in y
    Ixx = [] #second der in x
    Iyy = [] #second der in y 
    Id = [] #magnitude of the first der 
    Idd = [] #magnitude of the second der
    
    for ind in range(len(I)):
        Ix.append( imd[x[ind]+1,y[ind]] - imd[x[ind]-1,y[ind]] )
        Ixx.append( imd[x[ind]+1,y[ind]] - 2*imd[x[ind],y[ind]] + imd[x[ind]-1,y[ind]] )
        Iy.append( imd[x[ind],y[ind]+1] - imd[x[ind],y[ind]-1] )
        Iyy.append( imd[x[ind],y[ind]+1] - 2*imd[x[ind],y[ind]] + imd[x[ind],y[ind]-1] )
        Id.append(np.linalg.norm([Ix,Iy]))
        Idd.append(np.linalg.norm([Ixx,Iyy]))
    descriptor = np.array( list(zip(list(x),list(y),list(I),Ix,Iy,Ixx,Iyy,Id,Idd)),dtype='int64' ).T     # descriptors
    C = np.cov(descriptor) #covariance of the descriptor
    return C
def covd_ri(mat):
    ims = coo_matrix(mat)
    [x,y,I] = np.array([ims.row,ims.col,ims.data],dtype='int64')  
    dist = []
    nn1_ave = []
    nn1_std = []
    nn2_ave = []
    nn2_std = []
    nn3_ave = []
    nn3_std = []
    for ind in range(len(I)):
        dist.append( np.linalg.norm([x[ind],y[ind]]) )
        # the first layer of neightbors of the pixel #ind
        idx_nnx = np.argwhere( abs(x-x[ind])<=1 )
        idx_nny = np.argwhere( abs(y-y[ind])<=1 )
        idx_1nn = np.setdiff1d(np.intersect1d(idx_nnx,idx_nny),ind) #remove the presend ind pixel
        if len(idx_1nn) > 0:
            delta_1nn = I[idx_1nn]-I[ind] #radial difference of intensity; which equals the numerical second derivative
            nn1_ave.append(np.mean(delta_1nn))
        else:
            nn1_ave.append(0.0)
        if len(idx_1nn) > 1:
            nn1_std.append(np.std(delta_1nn))
        else:
            nn1_std.append(0.0)
        # the second layer of neightbors of the pixel #ind
        idx_nnx = np.argwhere( abs(x-x[ind])<=2 )
        idx_nny = np.argwhere( abs(y-y[ind])<=2 )
        center = np.union1d(ind,idx_1nn)
        idx_2nn = np.setdiff1d(np.intersect1d(idx_nnx,idx_nny),center)
        if len(idx_2nn) > 0:
            delta_2nn = I[idx_2nn]-I[ind] #radial difference of intensity; which equals the numerical second derivative
            nn2_ave.append(np.mean(delta_2nn))
        else:
            nn2_ave.append(0.0)
        if len(idx_2nn) > 1:
            nn2_std.append(np.std(delta_2nn))
        else:
            nn2_std.append(0.0)
        # the third layer of neightbors of the pixel #ind
        idx_nnx = np.argwhere( abs(x-x[ind])<=3 )
        idx_nny = np.argwhere( abs(y-y[ind])<=3 )
        center = np.union1d(center,idx_2nn)
        idx_3nn = np.setdiff1d(np.intersect1d(idx_nnx,idx_nny),center)
        if len(idx_3nn) > 0:
            delta_3nn = I[idx_3nn]-I[ind] #radial difference of intensity; which equals the numerical second derivative
            nn3_ave.append(np.mean(delta_3nn))
        else:
            nn3_ave.append(0.0)
        if len(idx_3nn) > 1:
            nn3_std.append(np.std(delta_3nn))
        else:
            nn3_std.append(0.0)

    descriptor=np.array( list(zip(list(I),dist,nn1_ave,nn2_ave,nn3_ave,nn1_std,nn2_std,nn3_std)),dtype='int64' ).T     # 8 features
    C = np.cov(descriptor) #covariance of the descriptor
    return C

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def viz_network_clusters(graph, partition, position, figname,min_cell_numb=10000,figsize=(50, 50), node_size=1, plot_overlaps=False,
                          plot_labels=False):
    """
    Plot a graph with node color coding for communities.
    :param graph: NetworkX/igraph graph
    :param partition: NodeClustering object
    :param position: A dictionary with nodes as keys and positions as values. Example: networkx.fruchterman_reingold_layout(G)
    :param figsize: the figure size; it is a pair of float, default (50, 50)
    :param node_size: int, default 1
    :param plot_overlaps: bool, default False. Flag to control if multiple algorithms memberships are plotted.
    :param plot_labels: bool, default False. Flag to control if node labels are plotted.
    Example:
    >>> from cdlib import algorithms, viz
    >>> import networkx as nx
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> pos = nx.spring_layout(g)
    >>> viz.plot_network_clusters(g, coms, pos)
    """
    partition = partition.communities
    graph = convert_graph_formats(graph, nx.Graph)

    n_communities = np.count_nonzero([len(c)>min_cell_numb for c in partition])
#     n_communities = len(partition)
    cmap = get_cmap(n_communities)
    
    plt.figure(figsize=figsize)
    plt.axis('off')

    for i in range(n_communities):
        if len(partition[i]) > min_cell_numb:
            COLOR = [cmap(i) for count in range(len(partition[i]))]
            if plot_overlaps:
                size = (n_communities - i) * node_size
            else:
                size = node_size
            fig = nx.draw_networkx_nodes(graph, position, node_size=size,nodelist=partition[i], node_color=COLOR)
        
    if plot_labels:
        nx.draw_networkx_labels(graph, position, labels={node: str(node) for node in graph.nodes()})

    plt.savefig(figname+'.tif',bbox_inches='tight',format='tiff')
    plt.close()

    return fig
def viz_network_single_clusters(graph, partition, position, figname,min_cell_numb=10000,figsize=(50, 50), node_size=1, plot_overlaps=False,
                          plot_labels=False):
    """
    Plot a graph with node color coding for communities.
    :param graph: NetworkX/igraph graph
    :param partition: NodeClustering object
    :param position: A dictionary with nodes as keys and positions as values. Example: networkx.fruchterman_reingold_layout(G)
    :param figsize: the figure size; it is a pair of float, default (50, 50)
    :param node_size: int, default 1
    :param plot_overlaps: bool, default False. Flag to control if multiple algorithms memberships are plotted.
    :param plot_labels: bool, default False. Flag to control if node labels are plotted.
    Example:
    >>> from cdlib import algorithms, viz
    >>> import networkx as nx
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> pos = nx.spring_layout(g)
    >>> viz.plot_network_clusters(g, coms, pos)
    """
    partition = partition.communities
    graph = convert_graph_formats(graph, nx.Graph)
    
    n_communities = np.count_nonzero([len(c)>min_cell_numb for c in partition])
    cmap = get_cmap(n_communities)
    
    for i in range(n_communities):
        if len(partition[i]) > min_cell_numb:
            plt.figure(figsize=figsize)
            plt.axis('off')
            COLOR = [cmap(i) for count in range(len(partition[i]))]
            if plot_overlaps:
                size = (n_communities - i) * node_size
            else:
                size = node_size
            fig = nx.draw_networkx_nodes(graph, position, node_size=size,nodelist=partition[i], node_color=COLOR)
        plt.savefig(figname+str(i)+'.png',bbox_inches='tight',format='png')
        plt.close()

    if plot_labels:
        nx.draw_networkx_labels(graph, position, labels={node: str(node) for node in graph.nodes()})

    return fig


def renormalization(K,omegaI,omegaIJ,m,threshold):
    #Find max btw node and edges:
    IJmax_idx = np.where( omegaIJ.data==np.max(omegaIJ.data[np.nonzero(omegaIJ.data)]) )[0][0] #idx of max nonzero edge
    IJmax = omegaIJ.data[IJmax_idx] #value of max nonzero edge
    i0 = np.where( omegaI==np.max(omegaI[np.nonzero(omegaI)]) )[0][0] #label of max nonzero node connectivity
    Imax = omegaI[i0][0,0] #value of max nonzero node connectivity
    maxtype = np.argmax([Imax,IJmax]) # max btw node and edge

    idx_i0isrow = np.argwhere(K.row==i0) # idxs of (i0,.)
    idx_i0iscol = np.argwhere(K.col==i0) # idx of (.,i0)
    js = np.unique(K.col[idx_i0isrow]) # nn j in (i0,.)
    connectivity = omegaI[i0]*m[i0]

    for i in js: #for any node in the nn of i0
        idx_ri = np.argwhere(K.row==i) #(i,.)
        idx_ci = np.argwhere(K.col==i) #(.,i)
        for j in js[np.argwhere(js==i)[0][0]+1:]:
            idx_cj = np.argwhere(K.col==j) #(.j)
            idx_rj = np.argwhere(K.row==j) #(j,.)
            idx_ij = np.intersect1d(idx_ri,idx_cj) #(i,j)
            idx_ji = np.intersect1d(idx_rj,idx_ci) #(j,i)
            idx_ii0 = np.intersect1d(idx_ri,idx_i0iscol) #(i,i0)   
            idx_i0j = np.intersect1d(idx_i0isrow,idx_cj) #(i0,j)
            if idx_ij.shape[0]>0: #update edge value if it exists, without the need to create new .row and .col
                K.data[idx_ij] = np.sum(np.append(K.data[idx_ij],K.data[idx_ii0]*K.data[idx_i0j]/connectivity)) #(i,j)
                K.data[idx_ji] = K.data[idx_ij] #(j,i)
                omegaIJ.data[idx_ij] = K.data[idx_ij]*(1.0/m[i]+1.0/m[j]) 
                omegaIJ.data[idx_ij] = omegaIJ.data[idx_ji]
            if idx_ij.shape[0]==0: #create a new edge
                newk = K.data[idx_ii0]*K.data[idx_i0j]/connectivity
                if newk >= 0.25*threshold: #set a threshold to avoid a dense graphs being generated in the flow
                    K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = expand1(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,newk,m,i,j)
    #remove i0 from K, omegaIJ
    K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.argwhere(K.row==i0))
    K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,np.argwhere(K.col==i0))
    #update omegaI
    for j in js: #for any node in the nn of i0
        omegaI[j] = sum([K.data[idx] for idx in np.argwhere(K.row==j)])*1.0/m[j]
    omegaI[i0] = 0.0  #set i0 to 0 in omegaI

    return K,omegaI,omegaIJ,Imax,IJmax
