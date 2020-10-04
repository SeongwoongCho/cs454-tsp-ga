import random
import numpy as np
import os

def seed_everything(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def calc_distance(x1,x2):
    ## x1,x2 : [a1,a2] 
    a1,a2 = x1
    b1,b2 = x2
    return ((a1-b1)**2 + (a2-b2)**2) ** 0.5

def parse_tsp(file):
    datas = []
    for i,line in enumerate(open(file, 'r')):
        if line == 'EOF':
            break
        if i>=6:
            line = line[:-1].split(' ')
            datas.append([float(line[1]),float(line[2])])    
    return datas

def fitness(order,DATAS):
    total_distance = 0
    for i in range(0,len(order)):
        total_distance += calc_distance(DATAS[order[i]], DATAS[order[i-1]])

    return total_distance