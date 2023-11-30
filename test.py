import random
from graph import Graph


my_graph = Graph([0,1,2,3,4,5])
liste_arcs = [(0,1,8),(0,5,10),(1,2,1),(2,5,-4),(2,3,-1),(3,4,-2),(4,5,1),(5,3,2)]
my_graph.add_edges(liste_arcs)

Graph.show_graph(my_graph)
path,dist,nb_iter,state = my_graph.search_bellman_ford(0,None)

print(f"Chemin : {path}")
print(f"Distance : {dist}")
print(f"Nombre d'itérations : {nb_iter}")

