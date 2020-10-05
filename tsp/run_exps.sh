#!/bin/bash

exps=(1)
greedy_ratios=(0 0.5 0.75 0.9 0.97)
num_clusters=(1 2 4 8 16)
#exps=(1 2)
#greedy_ratios=(0)
#num_clusters=(16)

cnt=0

for exp in ${exps[@]}; do
    for greedy_ratio in ${greedy_ratios[@]}; do
        for n_c in ${num_clusters[@]}; do
            python solver.py -t rl11849.tsp -n 20 -p 500 -g 500 -f 100000000 --elitism_rate 0.2 --offspring_rate 1 --init partial_greedy --crossover my --greedy_ratio ${greedy_ratio} --num_clusters ${n_c} --kmeans_iter 200 --save_dir ./logs/exp${cnt}/${exp}
            ((cnt=${cnt}+1))
        done
    done
done