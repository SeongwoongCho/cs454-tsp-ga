import numpy as np
import random

def pmx(p1,p2,**kwargs):
    spring = [-1] * len(p2)

    a = random.randint(1, len(p2) - 1)
    b = random.randint(a, len(p2))

    spring[a:b] = p2[a:b]

    for i in (list(range(a)) + list(range(b, len(p1)))):
        if not p1[i] in spring:
            spring[i] = p1[i]
        else:
            t = p2[i]
            while (t in spring) or (t not in p1[a:b]):
                t = p2[p1.index(t)]
            spring[i] = t
    return spring

def order(p1,p2,**kwargs):
    """
    suggested in the lecture note. crossover for sequences 
    """
    
    spring1 = p1.copy()
    spring2 = p2.copy()
    
    a = random.randint(1, len(p1) - 1)
    b = random.randint(a, len(p1))
    
    for e in spring1[a:b]:
        if e in spring2:
            spring2.remove(e)
    
    spring1[:a] = spring2[:a]
    spring1[b:] = spring2[a:]
               
    return spring1

def my(p1,p2):
    if np.random.uniform()<0.5:
        return pmx(p1,p2)
    else:
        return order(p1,p2)

def get_crossover(name,**kwargs):
    if name == 'pmx':
        return pmx
    if name == 'order':
        return order
    if name == 'my':
        return my