ordre = question_5(union_T)

def question_6(H, ordre):
    """
    Pour le graphe H généré à la question 3 appliquer l'algorithme Bellman-Ford en
    utilisant l'ordre <tot.
    """
    print("Question 6: Application de Bellman Ford à H")
    return H.bellman_ford(0,ordre)