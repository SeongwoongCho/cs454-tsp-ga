## 923288

## additional ideas
- stochastic greedy init?
    - 굳이 randomness를 줄필요 없이, 혹은 매우 적은 randomness를 주면서, n in [1,2,3] n-way greedy algorithm을 random하게 하면서 다양성을 확보

# method
python solver.py -t rl11849.tsp -n 20 -p 500 -g 500 -f 100000000 --elitism_rate 0.2 --offspring_rate 1 --init partial_greedy --crossover my --greedy_ratio 1 --num_clusters 16 --greedy_max_ways 2 --kmeans_iter 200 --save_dir ./logs/stochastic_greedy

# Methodology 
## concept : Good parents make a good offspring

## controlling dynamics of genetic algorithm using temparature
1. Genetic algorithm
    1. initialize population
        - designing : reasonable한 서칭 공간을 만들어 놓고, 해당 공간 사이의 조합을 최대화 시키는 것이 좋을 듯. 
        - random greey init : 처음에 random확률이 높은게 좋나, 나중에 random 확률이 높은게 좋나? 시간의 경우는 처음에 random확률이 높은게 좋음. random 확률 scheduling? mcmc, random, shortest 중애서 선택? 
        - mcmc init?
        - SA?
        - inversion sequence
    2. 
2. mutation scheduling?

## property

randomness = diversity
possible_search_space != total_search_space


greedy_ratio : 낮음 수록 randomness 커짐, possible_search_space 작아짐
elitism_rate : 낮을 수록 randomness 커짐
mutation_rate : 높을 수록 randomness 커짐
num_clusters : 높을 수록 possible_search_space 작아짐, randomness 작아짐



## bench_mark 

1. num_clusters, greedy_ratio 를 동시에 조절

population = 1000, generation = 1000, merge_g = step_scheduling

-> merge_g = (generation//(log_2(num_clusters)))

num_clusters vs greedy_ratio 조절하면서 performance, time-consuming 판단

num_clusters = [1,2,4,8,16]
greedy_ratio = [0,0.5,0.75,0.875,0.95,0.98,1]

각 실험 당 10회 씩 반복해서, 평균값을 2d plot, 물론 sigma도 기록함

fitness limit를 고정하고, fitness limit에 따라 generation을 정함..! 

5*10*10 = 500회 실험

## report

Title : clustered Genetic algorithms with partial greedy initialization

0. Summary(Conclusion) of report
    - 
1. Motivation
    - NP-hard problem like TSP has enormous search space where the number of cities increase. For GA, to
2. usage and explanation for flags
3. Implementation trick
    - Parallel GA by using python multiprocessing Pool
    - 
4. Algorithm Introduction
    1. 전체 도식표
    2. kmeans-clustering
    3. 
5. experimental results
6. conclusion
    - BO,GA for hyperparameter optimization 