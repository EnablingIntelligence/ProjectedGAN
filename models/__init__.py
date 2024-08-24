def kcal(n = 14):
    kidney = 330 * 5
    erbsen_karotten = 45 * 5.2
    mais = 90 * 2.85
    mushrooms = 30 * 4
    haferflocken = 340 * 2.5
    total = (kidney + erbsen_karotten + mais + mushrooms + haferflocken) 
    return total / n


print(kcal())