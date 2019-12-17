import igraph as ig
import leidenalg as la

G = ig.Graph.Famous('Zachary')
partition = la.find_partition(G, la.CPMVertexPartition,resolution_parameter = 0.05)
ig.plot(partition) # doctest: +SKIP
