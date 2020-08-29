#!/bin/bash

RUN=$1
DARKRUN=$2
        
sbatch -p upex -t 100 ./process_module.sh $RUN 0 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 1 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 2 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 3 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 4 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 5 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 6 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 7 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 8 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 9 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 10 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 11 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 12 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 13 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 14 $DARKRUN
sbatch -p upex -t 100 ./process_module.sh $RUN 15 $DARKRUN
