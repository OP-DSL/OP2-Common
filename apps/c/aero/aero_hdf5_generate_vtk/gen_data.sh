#!/bin/bash

echo -n "" > data/filenames.txt


for N in 50 #100 #200 300 500
do
    for M in 750 900 950 2000 #20 50 100 200 300 500
    #for M in 1000?
    do
        for C in 0.1 0.01 0.001 0.0001
        do
            FNAME="out_n${N}_m${M}_c${C}.txt"
            if [ ! -f data/$FNAME ]; then
                echo -n "Generating file $FNAME ... "
                #echo "about to hit: ./aero_cuda -n $N -m $M -c $C > data/$FNAME"
                ./aero_cuda -n $N -m $M -c $C > data/$FNAME
                echo "DONE"
            else
                echo "$FNAME file exists, skipping. "
            fi
            echo "$FNAME" >> data/filenames.txt            
        done
    done
done

python3 generate_res_graphs.py