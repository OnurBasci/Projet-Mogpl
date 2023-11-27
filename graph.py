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
        G = nx.Graph()
        # Ajout des sommets
        G.add_nodes_from(graph.list_vertex)
        # Ajout des arcs avec les poids
        for edge in graph.list_edges:
            u, v, weight = edge
            G.add_edge(u, v, weight=weight)
        # Positionnement des sommets
        pos = nx.spring_layout(G)
        # Dessiner les sommets
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=8)
        # Dessiner les arcs
        edge_labels = {(u, v): f'{weight}' for u, v, weight in graph.list_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.show()



    @staticmethod
    def unifiy_paths(path : list,path2 : list,path3 : list,list_vertex : list) -> 'Graph':
        """
        Fonction qui unifie les chemins pour créer un graph
        :param path: chemin 1
        :param path2: chemin 2
        :param path3: chemin 3
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
        for p_1,p_2,p_3 in zip(path,path2,path3):
            add_edge_from_path(p_1)
            add_edge_from_path(p_2)
            add_edge_from_path(p_3)
        # Création du graph
        new_graph = Graph(list_vertex)
        new_graph.add_edges(list_edges)
        # Retourne le graph
        return new_graph
    
    @staticmethod
    def generate_random_weights(graph : 'Graph'):
        """
        Fonction qui génère des poids aléatoires pour un graphe donnée
        :param graph: Un graphe orienté sans poids
        :return: Creation d'un nouveau graphe avec des poids aléatoirement générés
        """
        random_graph = deepcopy(graph)
        list_edges = []
        for vertex in random_graph.graph:
            for i, neighboor in enumerate(random_graph.graph[vertex]):
                random_weight = random.randint(-10, 10)
                list_edges.append((vertex, neighboor[0], random_weight))
                random_graph.graph[vertex][i] = (neighboor[0], random_weight)
        random_graph.list_edges = list_edges
        return random_graph
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
            # Pour chaque sommet dans l'ordre donné
            for vertex in vertex_order:
                # Si le sommet n'est pas encore accessible, on passe
                if self.distances[i-1,vertex] == np.inf:
                    continue
                # Sinon, on met à jour les distances pour chaque voisins 
                for neighbor, weight in self.graph[vertex]:
                    new_distance = self.distances[i-1,vertex] + weight
                    if new_distance < self.distances[i,neighbor]:
                        # On met à jour la distance du voisin
                        self.distances[i,neighbor] = new_distance
                        # On met à jour le prédecesseur du voisin
                        self.predecessors[i, neighbor] = vertex

            self.nb_iter += 1
            # # Si les distances n'ont pas changé, on arrête l'algorithme
            if np.array_equal(self.distances[i,:],self.distances[i-1,:]):
                break

        self.paths = self._reconstruct_path(source_vertex)
        return self.paths,self.distances,self.nb_iter
    
    def _reconstruct_path(self, source_vertex : int ) -> list:
        """
            Fonction qui reconstruit les chemins à partir des prédecesseurs
            :param source_vertex: sommet de départ
            :return: liste des chemins
        """

        paths : list = []
        # Pour chaque sommet
        for vertex in self.vertex_order:
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
                # On récupère le prédecesseur du sommet
                current_vertex = int(self.predecessors[self.nb_iter,current_vertex])
            path.append(source_vertex)
            paths.append(path[::-1])
        return paths

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
            if len(self.graph[vertex]) > 0 and len(self.get_precedent(vertex)) <= 0:
                sources.append(vertex)

        return sources

    def get_puits(self):
        """
            <Ajourter la description>
            :return: <Ajourter la description>
        """
        puits = []
        for vertex in self.graph.keys():
            if len(self.graph[vertex]) <= 0 and len(self.get_precedent(vertex)) > 0:
                puits.append(vertex)

        return puits

    def delete_vertex(self, vertex_to_delete : int):
        """
            <Ajourter la description>
            :param vertex_to_delete: <Ajourter la description>
            :return: <Ajourter la description>
        """
        #remove vertex
        print(self.graph[vertex_to_delete])
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
            return 0
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
            #à changer
            if len(graphe.graph.keys()) <= 1:
                s1.append(list(graphe.graph.keys())[0])
                graphe.graph = {}
            else:
                graphe.delete_vertex(u_max)
                s1.append(u_max)
        s1.extend(s2)
        return s1
