import random
import numpy as np
import time
from utils import *
from mutation import mutate
from selection import elitism_selection
from tqdm import tqdm
from multiprocessing.pool import Pool

def k_means_clustering(datas, num_clusters = 1, n_iter = 1000):    
    xs = []
    ys = []
    for x,y in datas:
        xs.append(x)
        ys.append(y)
        
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    centroids = []
    for _ in range(num_clusters):
        centroids.append((np.random.uniform(xmin,xmax),np.random.uniform(ymin,ymax)))
    
    for _ in tqdm(range(n_iter)):
        ## update points
        base_sets = [[] for _ in range(num_clusters)]
        for o,(ptx,pty) in enumerate(datas):
            min_dis = np.inf
            loc = 0
            for i, (cent_x, cent_y) in enumerate(centroids):
                dis = ((ptx-cent_x)**2 + (pty - cent_y)**2)**0.5
                if dis < min_dis:
                    min_dis = dis
                    loc = i
            base_sets[loc].append(o)
        
        ## update centroids
        for i in range(num_clusters):
            x_cent = 0
            y_cent = 0
            for o in base_sets[i]:
                x_cent += datas[o][0]/len(base_sets[i])
                y_cent += datas[o][1]/len(base_sets[i])
            centroids[i] = (x_cent,y_cent)
    return base_sets, centroids

def connect(chromosome1,chromosome2):
    i = random.randint(0,len(chromosome1)-1)
    
    new_chromosome = chromosome1[:i] + chromosome2 + chromosome1[i:]
    return new_chromosome

def merge_clusters(populations, centroids,base_sets, elitism_rate, datas):
    new_populations = []
    new_centroids = []
    new_base_sets = []
    while(len(populations) > 0):
        if len(populations) == 1:
            new_population = populations.pop(0)
            new_centroid = centroids.pop(0)
            new_base_set = base_sets.pop(0)
        else:
            # first, randomly select population, then find closest population using centroids.
            idx1 = random.randint(0,len(populations)-1)
            x1,y1 = centroids[idx1]
            min_dist = np.inf
            for idx in range(len(populations)):
                if(idx == idx1):
                    continue
                x2,y2 = centroids[idx]
                dist = ((x2-x1)**2 + (y2 - y1)**2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    idx2 = idx
            
            ## update new_centroid, new_base_set
            cent_x = 0
            cent_y = 0
            new_base_set = base_sets[idx1] + base_sets[idx2]
            for o in new_base_set:
                cent_x += datas[o][0]/len(new_base_set)
                cent_y += datas[o][1]/len(new_base_set)
            new_centroid = (cent_x,cent_y)
            
            base_sets.pop(max(idx1,idx2))
            base_sets.pop(min(idx1,idx2))
            centroids.pop(max(idx1,idx2))
            centroids.pop(min(idx1,idx2))
            
            #  second, merge two population into single population. new_population size = sum of each of two populations
            #    merging policy is selection -> connection(different from crossover)
            new_population = []
            new_popsize = len(populations[idx1])+len(populations[idx2])
            while len(new_population) < 5*new_popsize:
                chromosome1, fit1 = elitism_selection(populations[idx1], elitism_rate = elitism_rate)
                chromosome2, fit2 = elitism_selection(populations[idx2], elitism_rate = elitism_rate)
                
                new_chromosome = connect(chromosome1,chromosome2)
                fit = fitness(new_chromosome,datas)
                new_population.append((new_chromosome, fit))
            
            new_population = sorted(new_population, key=lambda x: x[1], reverse=False)
            new_population = new_population[:new_popsize]
            
            populations.pop(max(idx1,idx2))
            populations.pop(min(idx1,idx2))
            
        new_populations.append(new_population)
        new_centroids.append(new_centroid)
        new_base_sets.append(new_base_set)
    
    return new_populations, new_centroids, new_base_sets

def get_initialization(name):
    if name == 'random':
        return random_initialization
    if name == 'mcmc':
        return mcmc_initialization
    if name == 'greedy':
        return greedy_initialization
    if name == 'partial_greedy':
        return partial_greedy_initialization
    if name == 'weighted_partial_greedy':
        return weighted_partial_greedy_initialization

def random_initialization(population_size,num_workers,datas, base_set):
    return partial_greedy_initialization(population_size, num_workers, greedy_ratio = 0, datas = datas, base_set = base_set)

def greedy_initialization(population_size, num_workers, datas, base_set):
    return partial_greedy_initialization(population_size, num_workers, greedy_ratio = 1, max_ways = 1, datas = datas, base_set = base_set)

def weighted_partial_greedy_initialization(population_size, num_workers, greedy_weights, greedy_ratios, datas, base_set):
    """
    overselect에서 motivated. greedy_weights가 큰 것과 작은 것을 정해두고 각각의 비율만큼 뽑는다.
    """
    population = []
    for greedy_weight, greedy_ratio in zip(greedy_weights, greedy_ratios):
        _population = partial_greedy_initialization(int(population_size * greedy_weight), num_workers, greedy_ratio, 1, datas, base_set)
        population += _population
    return population

def partial_greedy_initialization(population_size, num_workers, greedy_ratio, max_ways, datas, base_set):
    pool = Pool(num_workers)
    population = pool.map(partial_greedy,[(datas,greedy_ratio,max_ways,base_set)]*population_size)
    pool.close()
    pool.join()
    return population

def partial_greedy(args):
    """
    주어진 도시들에서 greedy_ratio의 비율만큼 random subset을 뽑고 해당 지역들에 대하여 greedy algorithm을 수행
    """
    
    datas, greedy_ratio,max_ways,base_set = args
    random.shuffle(base_set)
    pivot = max(1,int((1-greedy_ratio) * len(base_set)))
    chromosome = base_set[:pivot].copy()
    new_base_set = base_set[pivot:].copy()
    
    assert max_ways in [1,2], "max_ways should be 1 or 2 for time complexity ~ n**(max_ways)"
    
    while len(new_base_set) > 0:
        if max_ways == 1 or np.random.uniform() <= 0.5 or len(new_base_set) == 1:
            current_node = chromosome[-1]
            min_dist = np.inf
            min_query_arg = 0
            for arg,query_node in enumerate(new_base_set):
                dist = calc_distance(datas[current_node], datas[query_node])
                if dist < min_dist:
                    min_dist = dist
                    min_query_arg = arg
            chromosome.append(new_base_set.pop(min_query_arg))
        else:
            current_node = chromosome[-1]
            min_dist = np.inf
            for arg1, query_node1 in enumerate(new_base_set):
                for arg2, query_node2 in enumerate(new_base_set):
                    if arg2 == arg1:
                        continue
                    else:
                        dist = calc_distance(datas[current_node],datas[query_node1]) + calc_distance(datas[query_node1],datas[query_node2])
                        
                        if dist < min_dist :
                            min_dist = dist
                            min_query_arg1 = arg1
                            min_query_arg2 = arg2
            if min_query_arg2 < min_query_arg1:
                query_node1 = new_base_set.pop(min_query_arg1)
                query_node2 = new_base_set.pop(min_query_arg2)
            else:
                query_node2 = new_base_set.pop(min_query_arg2)
                query_node1 = new_base_set.pop(min_query_arg1)
                
            chromosome.append(query_node1)
            chromosome.append(query_node2)
    
    fit = fitness(chromosome,datas)
    
    return chromosome, fit

"""
mcmc is deprecated

def mcmc_initialization(population_size, n_iter,T,num_workers, datas, base_set):
    pool = Pool(num_workers)
    population = pool.map(mcmc,[(n_iter, T, datas, base_set)]*population_size)
    
    pool.close()
    pool.join()
    return population

def mcmc(args):
    n_iter, T, datas,base_set = args
    chromosome = base_set.copy()
    random.shuffle(chromosome)

    fit = fitness(chromosome, datas)

    for j in range(n_iter):
        new_chromosome,new_fit = mutate(chromosome, datas)
        delta = new_fit - fit
        p = min(1, np.exp(-1 * delta / T)) # new_fit < fit 이면 accept, new_fit > fit 이면 확률적으로. 
        if np.random.uniform() < p:
            chromosome = new_chromosome
            fit = new_fit    
    return chromosome, fit
    

"""

if __name__ == '__main__':
    datas = parse_tsp('../rl11849.tsp')
    base_sets, centroids = k_means_clustering(datas,1, 1)
    initialize = get_initialization('greedy')
    st = time.time()
    population = initialize(20,1,datas,base_sets[0])
    print(time.time()-st)