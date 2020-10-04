# 1. Fast Genetic algorithm with Clustered greedy initialization
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
So, I focus on the initialization method which makes the superior initialized population. Specifically, My method is reducing search space using k-means clustering and making great parents using (partial) greedy algorithm. In the term (partial) greedy algorithm is conducted by simply doing greedy algorithm on the subset of the whole city, and subset size is controlled by greedy_ratio. ( greety_ratio 0 for the random initializaiton and 1 for the original greedy algorithm) Finally, I can control the balance by greedy_ratio and cluster numbers.
This problem(rl11849) has >10000 cities, > 10000! search space. Thus, clustering effectively reduces the search space and execution time. 

# 3. Algorithm and Implementation
## 3.1 Parallel Genetic Algorithm
I implement multiprocessing genetic algorithm using python 'multiprocessing' library.
I can boost the training speed by using 20 workers simultaneously.
Below is Example usage of multiprocessing

```
from multiprocessing import Pool

def func(arg):
  return arg

pool = Pool(num_workers)
## args is list of arg
l = pool.map(func, args)

```
I adjust this procedure on all the genetic procedure: initialization, selection, crossover, mutation

## 3.2 Algorithm
### 3.2.1 Basic Definition
### 3.2.2 Block Diagram of algorithm
The main difference between my algorithm and normal GA is that my algorithm has additional k-mean clustering block and cluster Merging block.

![image](https://github.com/SeongwoongJo/cs454-tsp-ga/blob/master/tsp/images/overall%20algorithm%20block%20diagram.png)

### 3.3.3 Step by Step code analysis
Selection, Mutation, Crossover is same as the general procedure. Thus in this part, I will explain the three components simply.
1. Initialization
- K-means clustering
- Partial greedy algorithm

2. GA procedure(Selection, Mutation, Crossover)
The GA procedure is done on each of the clusters. 
3. Merge clusters
![image](merging_algorithm_diagram.png)

# 4. Experimental results
The evaluation of the algorithm is viewed on two perspective: 1.Time and 2.Performance

## 4.1 Parallel GA

|  num workers  |   time        |    
| ------------- |:-------------:|
| 1  |  |
| 20 |  | 

The time is calculated on the condition of greedy initialization(when greedy_ratio = 1) with population_size = 20
We can observe that parallel GA with many workers can speedup the time tremendously.

## 4.2 Evolution Curve
![image](https://github.com/SeongwoongJo/cs454-tsp-ga/blob/master/tsp/images/evolution-curve.png)
The above curve is the example curve when num_clusters = 16, merge_g = 100.
On the every beginning iteration right after the merge, the distance rapidly decreases. You can observe that on every iteration around multiples of merge_g(=100), there is sudden improvement.
We can explain these phenomenons through the size of search space. When num_clusters is high before merge, each of clusters has small amount of cities and small search space. So, it can converge rapidly. On every merge step, it gradually increase possible search space and escape from the local minima.

## 4.3 Clustered greedy algorithm
### 4.3.1 Default setting
I run totally 25 experiments for the comparison. The final value can be changed according to the seed, but an approximate trend might be right.
- Running Experiments
```
./run_exps.sh
```

- Fixed Hyperparameters
```
num_workers : 20
population : 500
generation : 500
fitness_limit : 100000000 ## no limitness of fitness call
elitism_rate : 0.2
init : partial_greedy
crossover : my
kmeans_iter : 200
merge step : linearly scheduling (= max_generation//(np.log2(num_clusters)+1))
```

- Search Space

```
greedy_ratio : [0 0.5 0.75 0.9 0.97]
num_clusters : [1 2 4 8 16]
```

### 4.3.2 Hyperparameters-Performance result
![image](https://github.com/SeongwoongJo/cs454-tsp-ga/blob/master/tsp/images/hps-tsp_distance.png)

### 4.3.3 Hyperparameters-Time result
![image](https://github.com/SeongwoongJo/cs454-tsp-ga/blob/master/tsp/images/hps-time.png)

### 4.3.4 Time-Performance result
![image](https://github.com/SeongwoongJo/cs454-tsp-ga/blob/master/tsp/images/time-performance.png)

|         |   distance    | GA time  | init time |
| ------------- |:-------------:| -----:| ------:|
| Best-time exp | 1619243.71 | 6987.94 | 22.905 |
| Best-distance exp | 1100685      | 19100.5 | 1344.6 |

By comparing two points, one is best-time experiment and the other is best-distance experiment, the best-time experiment gains >270% time benefit with <50% distance loss.
On the restrict computing resource, K-means clustering with large K will be helpful by giving large time-benefit.


### 4.3 My Final Submission

# 5. Failed approaches
## 5.1 mcmc initialization
## 5.2 stochastic greedy initialization

# 6. Conclusion
시간, 퍼포먼스 둘다 좋음
