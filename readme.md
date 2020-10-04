# 1. Fast and Smart Genetic algorithm with Clustered greedy initialization
20170620 SeongwoongJo

Before start, I've never seen any other papers and references except for the slides from the lecture and 'pmx' algorithm.
All the implementations are made from me

# 2. Introduction
## 2.1 Key Idea
1. Parallel GA for python multiprocessing pool
2. Good parents make a good offspring. Let's focus on the initialization - Kmeans clustring + greedy algorithm
3. Large-scale experiments for testing 

## 2.2 Example usage and description

```
python solver.py -t rl11849.tsp -p 500 -f 100000000 -n 20 -g 500 \
--elitism_rate 0.2 --init partial_greedy --crossover my \
--greedy_ratio 1 --num_clusters 16 --kmeans_iter 500 --save_dir ./logs/results

-t : the location of .tsp file to load
-p : number of population
-f : maximum fitness function call
-g : generation 
-n : number of multiprocessing worker

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

## 2.3 Observation and Motivation
I think that it is very important to control the balance between randomness(diversity) and superiortiy of each population. For the fixed population size, diversity and superiority are a relationship of trade-off. For example, as we search for the large space, the points are getting sparse, which means that diversity is increasing and superiority of each chromosome is decreasing.
So, I focus on the initialization method which makes the superior initialized population. Specifically, My method is reducing search space using k-means clustering and making great parents using (partial) greedy algorithm. In the term (partial) greedy algorithm is conducted by simply doing greedy algorithm on the subset of the whole city, and subset size is controlled by greedy_ratio. ( greety_ratio 0 for the random initializaiton and 1 for the original greedy algorithm)
Finally, I can control the balance by greedy_ratio and cluster numbers.

# 3. Algorithm and Implementation
## 3.1 Parallel Genetic Algorithm
I implement multiprocessing genetic algorithm using python 'multiprocessing' library.
I can boost the training speed by using 20 workers simultaneously.

```
from multiprocessing
```

## 3.2 Algorithm
### 3.2.1 Definition
### 3.2.2 Block Diagram of algorithm
The main difference between my algorithm and normal GA is that my algorithm has additional k-mean clustering block and cluster Merging block.

K-means Clustering
Initialization
Selection
Crossover
Mutation
Cluster merging

각 단계 별 Time complexity랑 Fitness function calling 계산하기

### 3.3.3 Step by Step code analysis

# 4. Experimental results
The evaluation of the algorithm is viewed on two perspective: 1.Time and 2.Performance

## 4.1 Parallel GA

num workers = 1 vs num workers = 20 일때 Intialization time 비교

## 4.2 Clustered greedy algorithm
### 4.2.1 Default setting
고정된 하이퍼 파라미터 
변동된 하이퍼 파라미터
merge step 

### 4.2.2 Hyperparameters-Performance result
[image]

### 4.2.3 Hyperparameters-Time result
[image]

### 4.2.4 Time-Performance result
[image]

# 5. Failed approaches
1. mcmc initialization
2. stochastic greedy initialization

# 6. Conclusion
