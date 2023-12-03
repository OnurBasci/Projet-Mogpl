def question_5(union_T):
    """
     Appliquer l'algorithme GloutonFas avec comme entrée T et retourner un ordre <tot.
    """
    print("Question 5: Calcul de l'ordre pour le graphe d'union")
    ordre = union_T.glouton_fas()
    print(f"Question 5 \n {ordre=}")
    return ordre