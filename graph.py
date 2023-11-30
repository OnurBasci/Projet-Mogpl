import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

class Graph:
    """
        Class qui représente un graph orienté avec des arcs pondérés

        :method_static: 
            - show_graph(graph : 'Graph') -> None
            - unifiy_paths(path : list,path2 : list,path3 : list,list_vertex : list) -> 'Graph'
            - generate_random_weights(graph : 'Graph') -> 'Graph'

        :method: 
            - add_edges(liste_edges : list) -> None
            - get_precedent(target_vertex : int) -> list
            - search_bellman_ford(source_vertex : int , vertex_order : list = None) -> (list,np.ndarray,int)
            - reconstruct_path(source_vertex : int ) -> list
            - show_bellmanford_result() -> None
            - get_sources() -> list
            - get_puits() -> list
            - delete_vertex(vertex_to_delete : int) -> None
            - get_diff_enter_exit(vertex : int) -> int
            - glouton_fas() -> list
    """
        
    def __init__(self, list_vertex : list):
        # liste des sommets
        self.list_vertex : list = list_vertex
        # liste des arcs
        self.list_edges : list = None
        # un ordre des sommets 
        self.vertex_order : list = None
        # représentation du graph sous forme de listes d'adjacence
        self.graph : dict = {x : [] for x in self.list_vertex}
        # distances pour l'algorithm de Bellman-Ford
        self.distances : np.ndarray = None
        # prédecesseurs pour l'algorithm de Bellman-Ford
        self.paths : list = None
        # nombre d'itérations pour l'algorithm de Bellman-Ford
        self.nb_iter :int = 0

    """
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Fonction static de la classe                                             │  
    └──────────────────────────────────────────────────────────────────────────┘
    """
    @staticmethod
    def show_graph(graph : 'Graph'):
        """
            Fonction qui affiche un graph avec les poids des arcs et les sommets
            :param graph: graph à afficher
            :return: None
        """
        G = nx.DiGraph()
        # Ajout des sommets
        G.add_nodes_from(graph.list_vertex)
        # Ajout des arcs avec les poids
        new_list_edges = [(u,v) for u,v,w in graph.list_edges]
        G.add_edges_from(new_list_edges)
        # Positionnement des sommets
        pos = nx.spring_layout(G)
        # Dessiner les sommets
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=8)
        # Dessiner les arcs
        edge_labels = {(u, v): f'{weight}' for u, v, weight in graph.list_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.show()



    @staticmethod
    def unifiy_paths(paths : list,list_vertex : list) -> 'Graph':
        """
        Fonction qui unifie les chemins pour créer un graph
        :param paths: chemins des graphes
        :param list_vertex: liste des sommets
        :return: un graph
        """
        # Initialisation des variables
        list_edges : list = []
        added_edges : dict = {}
        # Fonction qui ajoute un arc au graph si il n'existe pas déjà
        def add_edge_from_path(path):
            if len(path) <= 1:
                return
            for i in range(0,len(path)-1):
                vertex_1,vertex_2 = path[i],path[i+1]
                if added_edges.get((vertex_1,vertex_2),None) is not None:
                    continue
                list_edges.append((path[i],path[i+1],1))
                added_edges[(vertex_1,vertex_2)] = True

        # Ajout des arcs pour chaque chemin
        for p_x in zip(*paths):
            for p in p_x :
                add_edge_from_path(p)
        # Création du graph
        new_graph = Graph(list_vertex)
        new_graph.add_edges(list_edges)
        # Retourne le graph
        return new_graph
    
    @staticmethod
    def generate_graphs_with_random_weights(base_graph : 'Graph', nb_graph:int, delete_cycle : bool= True)->list['Graph']:
        """
        Fonction qui génère des graphes avec des poids aléatoires
        :param base_graph: Un graphe orienté sans poids
        :param nb_graph: nombre de graph à générer
        :return: liste des graphes générés
        """
        graphs = []
        for _ in range(nb_graph):
            graphs.append(Graph.generate_random_weights(base_graph, delete_cycle=delete_cycle))
        return graphs

    @staticmethod
    def generate_random_weights(graph : 'Graph', delete_cycle = True):
        """
        Fonction qui génère des poids aléatoires pour un graphe donnée
        :param graph: Un graphe orienté sans poids
        :return: Creation d'un nouveau graphe avec des poids aléatoirement générés
        """

        #generate random graph
        random_graph = deepcopy(graph)
        list_edges = []
        for vertex in random_graph.graph:
            for i, neighboor in enumerate(random_graph.graph[vertex]):
                random_weight = random.randint(-10, 10)
                list_edges.append((vertex, neighboor[0], random_weight))
                random_graph.graph[vertex][i] = (neighboor[0], random_weight)
        random_graph.list_edges = list_edges


        if delete_cycle:
            random_graph.delete_cycles()

        return random_graph

    
    @staticmethod
    def generate_random_order(graph):
        """
        Fonction qui génère un ordre aléatoire pour un graphe donnée
        """
        nb_elem = len(graph.list_vertex)
        return random.sample(range(nb_elem), nb_elem)
    
    @staticmethod
    def generate_random_graph(size_graph : int,nb_edges : int) -> 'Graph':
        """
            Fonction qui génère un graph aléatoire d'une taile donnée
        """
        liste_vertex : list = [i for i in range(size_graph)]
        liste_edges = [tuple(random.sample(liste_vertex,2)+[1]) for _ in range(nb_edges)]
        new_graph = Graph(liste_vertex)
        new_graph.add_edges(liste_edges)
        new_graph = Graph.generate_random_weights(new_graph)
        return new_graph

    @staticmethod
    def generate_level_graph(nb_level):
        """
        This function creates a graph by levels as explained in the guestion 11
        """
        level_graph = Graph([i for i in range(nb_level * 4)])

        edges = []
        for i in range(nb_level-1):
            next_level_vertexes = [(i*4)+4+k for k in range(4)]
            for j in range(4):
                for k in next_level_vertexes:
                    edges.append(((i*4)+j, k, 1))
        level_graph.add_edges(edges)
        level_graph = Graph.generate_random_weights(level_graph, delete_cycle=False)

        return level_graph

    
    @staticmethod
    def generate_compare_graph(size_graph:int,nb_edges:int,nb_graph_to_generate:int, delete_cycle = True):
        """
            Fonction qui compare les résultats de l'ordre glouton_fas avec un ordre aléatoire
            :param size_graph: taille du graph
            :param nb_edges: nombre d'arcs
            :param nb_graph_to_generate: nombre de graph à générer
        """
        source = 0
        # Génération d'un graph aléatoire
        graph = Graph.generate_random_graph(size_graph=size_graph,nb_edges=nb_edges)
        # Génération de graphes avec des poids aléatoires
        train_test_graphs = Graph.generate_graphs_with_random_weights(graph,nb_graph_to_generate, delete_cycle=delete_cycle)
        # Récupération des graphes d'entrainement et de test
        train_graphs = train_test_graphs[:nb_graph_to_generate-1]
        test_graph = train_test_graphs[nb_graph_to_generate-1]
        # Calcul de l'arborecence 
        paths = [graph.search_bellman_ford(source,None)[0] for graph in train_graphs]
        # Unifier les arborecences
        union_graph = Graph.unifiy_paths(paths,graph.list_vertex)
        # Calcul de l'ordre avec glouton_fas
        order = union_graph.glouton_fas()
        # Génération d'un ordre aléatoire
        random_order = Graph.generate_random_order(graph)
        #delete cycles caused by orders
        test_graph.delete_cycles(order)
        test_graph.delete_cycles(random_order)
        # Calcul de l'arborecence avec l'ordre glouton_fas
        #Graph.show_graph(test_graph)
        _,_,nb_iter_glouton,_ = test_graph.search_bellman_ford(source,order)
        _,_,nb_iter_random,_ = test_graph.search_bellman_ford(source,random_order)

        """
        print(f"Ordre glouton_fas: {order}")
        print(f"Nombre d'itérations avec glouton_fas: {nb_iter_glouton}")
        print(f"Ordre aléatoire: {random_order}")
        print(f"Nombre d'itérations avec un ordre aléatoire: {nb_iter_random}")"""

        return nb_iter_glouton, nb_iter_random

    @staticmethod
    def compare_graph(graph, nb_graph_to_generate):
        source = 0
        # Génération de graphes avec des poids aléatoires
        train_test_graphs = Graph.generate_graphs_with_random_weights(graph, nb_graph_to_generate,delete_cycle=False)
        train_graphs = train_test_graphs[:nb_graph_to_generate - 1]
        test_graph = train_test_graphs[nb_graph_to_generate - 1]
        # Calcul de l'arborecence
        paths = [graph.search_bellman_ford(source, None)[0] for graph in train_graphs]
        # Unifier les arborecences
        union_graph = Graph.unifiy_paths(paths, graph.list_vertex)
        # Calcul de l'ordre avec glouton_fas
        order = union_graph.glouton_fas()
        # Génération d'un ordre aléatoire
        random_order = Graph.generate_random_order(graph)
        # delete cycles caused by orders
        test_graph.delete_cycles(order)
        test_graph.delete_cycles(random_order)
        # Calcul de l'arborecence avec l'ordre glouton_fas
        # Graph.show_graph(test_graph)
        _, _, nb_iter_glouton, _ = test_graph.search_bellman_ford(source, order)
        _, _, nb_iter_random, _ = test_graph.search_bellman_ford(source, random_order)

        """
        print(f"Ordre glouton_fas: {order}")
        print(f"Nombre d'itérations avec glouton_fas: {nb_iter_glouton}")
        print(f"Ordre aléatoire: {random_order}")
        print(f"Nombre d'itérations avec un ordre aléatoire: {nb_iter_random}")"""

        return nb_iter_glouton, nb_iter_random

    """
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Fonction de la classe                                                    │
    └──────────────────────────────────────────────────────────────────────────┘
    """
    def add_edges(self, liste_edges : list) -> None:
        """
            Fonction qui ajoute des arcs au graph
            :param liste_edges: liste des arcs
            :return: None
        """
        self.list_edges = liste_edges
        for u,v,w in liste_edges:
            self.graph[u].append((v, w))

    def get_precedent(self, target_vertex : int) -> list:
        """
            Fonction qui retourne la liste des prédecesseurs d'un sommet
            :param target_vertex: sommet cible
            :return: liste des prédecesseurs du sommet cible

        """
        liste_precedent : list = []
        # Pour chaque sommet
        for vertex in self.graph.keys():
            # Pour chaque arc du sommet
            for v,w in self.graph[vertex]:
                # Si l'arc arrive sur le sommet cible, on l'ajoute à la liste des prédecesseurs
                if v == target_vertex:
                    liste_precedent.append((vertex,w))
        return liste_precedent

    def search_bellman_ford(self, source_vertex : int , vertex_order : list = None) -> (list,np.ndarray,int):
        """
            Fonction qui calcule le plus court chemin entre deux sommets avec l'algorithme de Bellman-Ford
            :param source_vertex: sommet de départ
            :param vertex_order: ordre des sommets à parcourir
            :return: liste des plus court chemins, matrice des distances, nombre d'itérations
        """

        # si vertex_order n'est pas spécifié, on prend l'ordre de la liste des sommets
        if vertex_order is None:
            vertex_order = self.list_vertex

        self.vertex_order = vertex_order
        # initialisation des variables
        self.nb_iter = 0
        nb_vertices = len(vertex_order)
        # initialisation des distances et des prédecesseurs
        self.distances = np.zeros((nb_vertices-1,nb_vertices))
        self.distances.fill(np.inf)
        self.distances[:,source_vertex] = 0
        self.predecessors = np.zeros(self.distances.shape)-1

        # Max itération de l'algorithme de Bellman-Ford
        for i in range(1, len(vertex_order)-1):
            self.distances[i] = self.distances[i-1]
            self.predecessors[i] = self.predecessors[i-1]
            # Pour chaque sommet dans l'ordre donné
            for vertex in vertex_order:
                for neighbor, weight in self.graph[vertex]:
                    new_distance = self.distances[i-1,vertex] + weight
                    if new_distance < self.distances[i-1,neighbor]:
                        # On met à jour la distance du voisin
                        self.distances[i,neighbor] = new_distance
                        # On met à jour le prédecesseur du voisin
                        self.predecessors[i, neighbor] = vertex

            self.nb_iter += 1
            # # Si les distances n'ont pas changé, on arrête l'algorithme
            if np.array_equal(self.distances[i,:],self.distances[i-1,:]):
                break

        pre_paths,state = self._reconstruct_path(source_vertex, vertex_order)
        if state:
            self.paths = pre_paths
            return self.paths,self.distances,self.nb_iter,True
        else:
            cycle = pre_paths
            return cycle,None,None,False

    
    def _find_negative_cycle(self, path : list) -> int:
        """
            Fonction qui trouve un cycle à poids négatif
            :param path: chemin
            :return: nombre d'itérations
        """
        already_seen = {}
        index_cycle = -1
        for i,value in enumerate(path):
            if value in already_seen:
                index_cycle = i
                break
            already_seen[value] = i

        i_start_cycle = already_seen[path[index_cycle]]
        cycle = path[i_start_cycle:index_cycle]    
        return cycle

    def _reconstruct_path(self, source_vertex : int, vertex_order=None) -> list:
        """
            Fonction qui reconstruit les chemins à partir des prédecesseurs
            :param source_vertex: sommet de départ
            :return: liste des chemins
        """
        MAX_LENGTH = len(self.list_vertex)-1
        paths : list = []
        # Pour chaque sommet
        for vertex in vertex_order:
            # Si le sommet n'est pas accessible, on passe
            if self.distances[self.nb_iter,vertex] == np.inf:
                continue
            # Sinon, on reconstruit le chemin
            path : list = []
            current_vertex = vertex
            # Tant qu'on est pas arrivé au sommet de départ
            while current_vertex != source_vertex:
                # On ajoute le sommet au chemin
                path.append(current_vertex)
                if len(path) > MAX_LENGTH:
                    cycle = self._find_negative_cycle(path)
                    return cycle,False

                # On récupère le prédecesseur du sommet
                current_vertex = int(self.predecessors[self.nb_iter,current_vertex])
            path.append(source_vertex)
            paths.append(path[::-1])
        return paths,True


    def show_bellmanford_result(self):
        """
            Fonction qui affiche les résultats de l'algorithme de Bellman-Ford
            :return: None
        """
        print("=========================\n")	
        print(f"[INFO] BellmanFord completed in {self.nb_iter} iterations")
        print(f"[INFO] Paths found : {self.paths}")
        print(f"[INFO] Matrix distance : \n {self.distances}")
        print("\n")

    def get_sources(self):
        """
            <Ajourter la description>
            :return: <Ajourter la description>
        """
        sources = []
        for vertex in self.graph.keys():
            if len(self.get_precedent(vertex)) <= 0:
                sources.append(vertex)

        return sources

    def get_puits(self):
        """
            <Ajourter la description>
            :return: <Ajourter la description>
        """
        puits = []
        for vertex in self.graph.keys():
            if len(self.graph[vertex]) <= 0:
                puits.append(vertex)

        return puits

    def delete_vertex(self, vertex_to_delete : int):
        """
            <Ajourter la description>
            :param vertex_to_delete: <Ajourter la description>
            :return: <Ajourter la description>
        """
        #remove vertex
        del self.graph[vertex_to_delete]
        #remove precedents
        for vertex in self.graph.keys():
            for arcs in self.graph[vertex]:
                if arcs[0] == vertex_to_delete:
                    self.graph[vertex].remove(arcs)

    def get_diff_enter_exit(self, vertex : int):
        """
            <Ajourter la description>
            :param vertex: <Ajourter la description>
            :return: <Ajourter la description>
        """
        if not(vertex in self.graph.keys()):
            return -math.inf
        sum_enter = sum(precedent[1] for precedent in self.get_precedent(vertex))
        sum_exit = sum(neighbor[1] for neighbor in self.graph[vertex])
        return sum_exit - sum_enter

    def glouton_fas(self):
        """
            <Ajourter la description>
            :return: <Ajourter la description>
        """

        s1 : list = []
        s2 : list = []
        graphe = deepcopy(self)
        while len(graphe.graph.keys()) > 0:
            while len(graphe.get_sources()) > 0:
                u = graphe.get_sources()[0]
                s1.append(u)
                graphe.delete_vertex(u)
            while len(graphe.get_puits()) > 0:
                u = graphe.get_puits()[0]
                s2.insert(0, u)
                graphe.delete_vertex(u)
            u_max = np.argmax(np.array([graphe.get_diff_enter_exit(vertex) for vertex in self.graph.keys()]))

            if len(graphe.graph.keys()) > 0:
                s1.append(u_max)
                graphe.delete_vertex(u_max)
        s1.extend(s2)
        return s1

    def delete_cycles(self, vertex_order=None):
        """
        remove negatif cycles until there is none
        """
        accepted = False
        while not accepted:

            cycle, _, _, state = self.search_bellman_ford(0, vertex_order)
            if state:
                accepted = True
            else:
                # print("Cycle négatif détecté, re-génération des poids")
                # print(f"Cycle: {cycle}")
                self.delete_negatif_cyle(cycle)

    def delete_negatif_cyle(self, cycle):
        """
        set all weights of edges between the vertex of the cycle and it's neighboors to positif
        """
        for k, bad_vertex in enumerate(cycle):
            for i, neighboor in enumerate(self.graph[bad_vertex]):
                # print(f"bad weight: {random_graph.graph[bad_vertex][i][1]}")
                v, w = self.graph[bad_vertex][i]
                self.graph[bad_vertex][i] = (v, abs(w))