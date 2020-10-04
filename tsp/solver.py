import numpy as np
import time
import argparse
import sys
import copy
import json
from tqdm import tqdm
from utils.utils import *
from utils.initialization import *
from utils.mutation import *
from utils.crossover import *
from utils.selection import *
from multiprocessing.pool import Pool

class Logger():
    def __init__(self, save_dir):
        os.makedirs(save_dir,exist_ok=True)
        self.save_dir = save_dir
    def write_args(self,args):
        with open(os.path.join(self.save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        f.close()
    def write_init(self, time):
        with open(os.path.join(self.save_dir,'logs.txt'), 'a+') as f:
            f.write("init time, {} \n".format(time))
            f.close()
    def write_gen(self,generation, time, best_value):
        with open(os.path.join(self.save_dir,'logs.txt'), 'a+') as f:
            f.write("{}, {}, {} \n".format(generation, time, best_value))
            f.close()
        
def get_args():
    parser = argparse.ArgumentParser('CS454 TSP implementation by seongwoongJo')
    parser.add_argument('-t', '--tsp', type=str, default='rl11849.tsp', help='the location of .tsp file to parse')
    parser.add_argument('-n', '--num_workers', type=int, default=20, help='num workers for parallel GA')
    parser.add_argument('-p', '--population', type = int, default = 1)
    parser.add_argument('-g', '--generation', type = int, default = 100)
    parser.add_argument('-f', '--fitness_limit', type = int, default = 2000)
    parser.add_argument('--elitism_rate', type = float, default = 0.1)
    parser.add_argument('--offspring_rate', type = float , default = 1., help = "the rate between parents and offsprings, rate = offsprings / parents")
    parser.add_argument('--init',type=str,default='random')
    parser.add_argument('--crossover', type = str, default = 'my')
    
    parser.add_argument('--greedy_ratio', type = float, default = 0.5)
    parser.add_argument('--greedy_weights', type = list, default = [0.1,0.9])
    parser.add_argument('--greedy_ratios', type = list, default = [0.9,0.1])
    parser.add_argument('--greedy_max_ways', type = int, default = 1)
    
    parser.add_argument('--num_clusters',type = int, default = 1)
    parser.add_argument('--kmeans_iter', type = int, default = 1000)
#    parser.add_argument('--merge_g', type = int, default = 200)
    parser.add_argument('--save_dir', type = str, default = './logs/')
    
    """
    ## below arguments are deprecated ##
    
    parser.add_argument('--debug', action = 'store_true')
    parser.add_argument('--mcmc_iter_ratio', type = float, default = 0.2, help = "the ratio between mcmc iteration and maximum fitness call ")
    parser.add_argument('--mcmc_temparature', type = float, default = 2e-8)
    """
    
    args = parser.parse_args()
    return args

def make_solution(save_dir,l):
    # l starts from 0-index, should be converted into 1-index
    f = open(os.path.join(save_dir,'solution.csv'), 'w', encoding='utf-8')
    for city in l:
        f.write(str(city+1)+'\n')
    f.close()
    return

def get_single_offspring(args):
    population, elitism_rate, datas, crossover, mutate = args
    parent1 = elitism_selection(population, elitism_rate)
    parent2 = elitism_selection(population, elitism_rate)
    offspring = crossover(parent1[0], parent2[0])
    offspring = mutate(offspring,datas)
    return offspring

if __name__ == '__main__':
    args = get_args()
    
    assert args.population * args.elitism_rate // args.num_clusters >= 2
    logger = Logger(args.save_dir)
    logger.write_args(args)
    datas = parse_tsp(args.tsp)
    
    start_time = time.time()
    
    base_sets, centroids = k_means_clustering(datas,args.num_clusters, args.kmeans_iter)
    crossover = get_crossover(args.crossover)
    initialize = get_initialization(args.init)
    
    populations = []
    if args.init == 'random' or args.init == 'greedy':
        for base_set in tqdm(base_sets):
            population = initialize(args.population//args.num_clusters,args.num_workers, datas,base_set)
            populations.append(population)
    elif args.init == 'partial_greedy':
        for base_set in tqdm(base_sets):
            population = initialize(args.population//args.num_clusters, args.num_workers, args.greedy_ratio, args.greedy_max_ways, datas,base_set)
            populations.append(population)
    elif args.init == 'weighted_partial_greedy':
        for base_set in tqdm(base_sets):
            population = initialize(args.population//args.num_clusters, args.num_workers, args.greedy_weights, args.greedy_ratios, datas,base_set)
            populations.append(population)
#    elif args.init == 'mcmc':
#        for base_set in tqdm(base_sets):
#            population, n_fitness_call = initialize(args.population//args.num_clusters, int(args.mcmc_iter_ratio * args.fitness_limit), args.mcmc_temparature, args.num_workers,datas,base_set)
#            total_fitness += n_fitness_call
#            populations.append(population)

    one_gen_fitness_call = int(args.offspring_rate*args.population) + 1
    max_generation = min((args.fitness_limit -  args.population) // one_gen_fitness_call, args.generation)
    total_fitness_call =  args.population + one_gen_fitness_call * max_generation
    merge_g = int(max_generation//(np.log2(args.num_clusters)+1))
    
    logger.write_init(time.time()- start_time)
    best = ([],np.inf)
    for g in range(max_generation):
        start_time = time.time()
        new_populations = []
        for i,population in enumerate(populations):
            mp_pool = Pool(args.num_workers)
            pool_args =(population, args.elitism_rate, datas, crossover, mutate)
            pool = mp_pool.map(get_single_offspring,[pool_args]*int(args.offspring_rate*args.population//len(populations)))
            mp_pool.close()
            mp_pool.join()
            
            pool += population
            pool = sorted(pool, key = lambda x : x[1], reverse = False)
            new_populations.append(pool[:args.population//len(populations)])
        
        populations = new_populations
        
        ## merge clustered populations
        merge_g = 5
        if len(base_sets) > 1 :
            if (g+1)%merge_g == 0:
                populations, centroids, base_sets = merge_clusters(populations, centroids,base_sets,args.elitism_rate, datas)
        
        ## nearest centroid approximation
        approx_best = populations[0][0][0].copy()
        curr_centroid_idx = 0
        remain_centroid_idxs = list(range(1,len(populations)))
        while len(remain_centroid_idxs) > 0:
            curr_x,curr_y = centroids[curr_centroid_idx]
            min_dist = np.inf
            
            for idx in remain_centroid_idxs:
                x,y = centroids[idx]
                dist = ((curr_x-x)**2 + (curr_y-y)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    next_centroid_idx = idx
            approx_best.extend(populations[next_centroid_idx][0][0].copy())
            remain_centroid_idxs.remove(next_centroid_idx)
            curr_centroid_idx = next_centroid_idx
            
        approx_best = (approx_best, fitness(approx_best, datas))
        
        if approx_best[1] < best[1]:
            best = approx_best
            
        print(best[1])
        logger.write_gen(g, time.time()-start_time, best[1])
    
    make_solution(args.save_dir,best[0])
    print(best[1])
