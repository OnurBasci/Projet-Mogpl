def question_8(Bellman_H, Bellman_H_random):
    """
     Comparer les résultats (nombre d'itérations) obtenus dans les questions 6 et 7.
    """
    print("Comparaison Bellman avec prétraitement et Bellman avec un ordre aléatoire")
    if not(Bellman_H) or not(Bellman_H_random):
        return
    print(f"Nb itération, Bellman avec prétraitement: {Bellman_H[2]}")
    print(f"Nb itération, Bellman avec un ordre aléatoire: {Bellman_H_random[2]}")
    print("[INFO] Question 8 terminée\n")