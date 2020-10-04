# Topic
clustered Genetic algorithms with greedy initialization

20170620 SeongwoongJo

# Key Idea
1. Parallel GA for python multiprocessing pool
2. Good parents make a good offspring. Let's focus on the initialization
3. Large-scale experiments for testing 

# example usage and option description

```
python solver.py -t rl11849.tsp -p 500 -f 100000000 -n 20 -g 500 \
--elitism_rate 0.2 --init partial_greedy --crossover my \
--greedy_ratio 1 --num_clusters 16 --kmeans_iter 500 --save_dir ./logs/results

-t : the location of .tsp file to load
-p : number of population
-f : maximum fitness function call
-g : generation 

--elitism_rate : elitism_rate for the maximum population
--init : the mode of initialization
--crossover : the mode of crossover

--greedy_ratio : Percentage of cities to apply the greedy algorithm
--num_clusters : initial number of the clusters
--kmeans_iter : k-means clustering iterations
--save_dir : where to save logs and hyperpameter informations

=====below arguments are deprecated(not used anymore)=====

****TODO

```

# Observation and Motivation
I think that it is very important to control the balance between randomness(diversity) and superiortiy of each population. For the fixed population size, diversity and superiority are a relationship of trade-off. For example, as we search for the large space, the points are getting sparse, which means that diversity is increasing and superiority of each chromosome is decreasing.
So, I focus on the initialization method which makes the superior initialized population. Specifically, My method is reducing search space using k-means clustering and making great parents using (partial) greedy algorithm. In the term (partial) greedy algorithm is conducted by simply doing greedy algorithm on the subset of the whole city, and subset size is controlled by greedy_ratio. ( greety_ratio 0 for the random initializaiton and 1 for the original greedy algorithm)
Finally, I can control the balance by greedy_ratio and cluster numbers.

# Algorithm and Implementation
## Parallel GA


## Algorithm
1. defination

2. Initialization
3. 

# experiments 
I num_clusters, greedy_ratio 


1. num_clusters, greedy_ratio 를 동시에 조절

population = 1000, generation = 1000, merge_g = step_scheduling

-> merge_g = (generation//(log_2(num_clusters)))

num_clusters vs greedy_ratio 조절하면서 performance, time-consuming 판단

num_clusters = [1,2,4,8,16]
greedy_ratio = [0,0.5,0.75,0.875,0.95,0.98,1]

각 실험 당 10회 씩 반복해서, 평균값을 2d plot, 물론 sigma도 기록함

fitness limit를 고정하고, fitness limit에 따라 generation을 정함..! 

5*10*10 = 500회 실험

# Failed approaches
1. mcmc initialization
2. stochastic greedy initialization


## report

Title : clustered Genetic algorithms with greedy initialization

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
    
