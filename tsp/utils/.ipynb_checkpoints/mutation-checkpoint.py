import numpy as np
import random
from .utils import *

def mutate(sequence, datas):
    new_sequence = sequence.copy()
    i,j = random.sample(range(len(sequence)),2)
    if i > j:
        i,j = j,i
        
    prob = np.random.uniform()
    if  prob< 0.05:
        ## random window switching
        new_sequence = new_sequence[:i] + new_sequence[j:] + new_sequence[i:j]
    elif prob < 0.1:
        ## random two city switching
        new_sequence[i],new_sequence[j] = new_sequence[j],new_sequence[i]
    fit = fitness(new_sequence,datas)
    return new_sequence, fit