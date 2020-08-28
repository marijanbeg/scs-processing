#!/bin/bash

RUN=$1
        
sbatch -p upex -t 10 ./process_module.sh $RUN 0
sbatch -p upex -t 10 ./process_module.sh $RUN 1
sbatch -p upex -t 10 ./process_module.sh $RUN 2
sbatch -p upex -t 10 ./process_module.sh $RUN 3
sbatch -p upex -t 10 ./process_module.sh $RUN 4
sbatch -p upex -t 10 ./process_module.sh $RUN 5
sbatch -p upex -t 10 ./process_module.sh $RUN 6
sbatch -p upex -t 10 ./process_module.sh $RUN 7
sbatch -p upex -t 10 ./process_module.sh $RUN 8
sbatch -p upex -t 10 ./process_module.sh $RUN 9
sbatch -p upex -t 10 ./process_module.sh $RUN 10
sbatch -p upex -t 10 ./process_module.sh $RUN 11
sbatch -p upex -t 10 ./process_module.sh $RUN 12
sbatch -p upex -t 10 ./process_module.sh $RUN 13
sbatch -p upex -t 10 ./process_module.sh $RUN 14
sbatch -p upex -t 10 ./process_module.sh $RUN 15
