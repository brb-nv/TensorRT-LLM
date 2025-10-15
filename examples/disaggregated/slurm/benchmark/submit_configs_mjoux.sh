#!/bin/bash

# a config consists of TP, CP, EP
configs="8,1,2 16,1,4 32,1,8 1,8,2 1,16,4 1,32,8 2,4,2 2,8,4 2,16,8"
ctx_len_start=$((2 ** 16))
ctx_len_end=$((2 ** 21))

for config in $configs; do
    IFS=","
    set -- $config
    tp=$1
    cp=$2
    ep=$3
    echo "TP $tp, CP $cp, EP $ep"
    ctx_len=$ctx_len_start
    while [ $ctx_len -le $ctx_len_end ]; do
        if [[ $ctx_len -ge $((2 ** 19)) && $cp -eq 1 ]]; then
            echo "skipping ctx_len $ctx_len because it is too long for cp 1"
        else
            sed -i "s/isl=[[:digit:]]\+/isl=$ctx_len/" submit_mjoux.sh
            sed -i "s/gen_tp_size=[[:digit:]]\+/gen_tp_size=$tp/" submit_mjoux.sh
            sed -i "s/gen_cp_size=[[:digit:]]\+/gen_cp_size=$cp/" submit_mjoux.sh
            sed -i "s/gen_ep_size=[[:digit:]]\+/gen_ep_size=$ep/" submit_mjoux.sh
            bash submit_mjoux.sh
        fi
        ctx_len=$((ctx_len * 2))
    done
done
