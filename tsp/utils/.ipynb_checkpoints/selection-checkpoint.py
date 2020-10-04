import random

def elitism_selection(population, elitism_rate):
    """
    population : List of (chromosome, fit)
    elitism_rate : 
    
    return (chromosome, k)
    """
    elitism_k = int(len(population)*elitism_rate)
    tournament_pool = random.sample(population, elitism_k)
    result = sorted(tournament_pool, key=lambda x: x[1], reverse=False)
    return result[0]