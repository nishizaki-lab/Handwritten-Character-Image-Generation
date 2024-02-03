#!/bin/bash

# python3 train.py -g 0 --pretraine &> experiment/train_etl.log &
# wait

# for gen_dataset in style_transfer_adain_with_emb_font2hand cdiff_select;
# do
#     python3 train.py -g 0 --use_gen ../data_paths/gen/${gen_dataset} --pretraine &> experiment/train_etl_${gen_dataset}.log &
#     wait
# done

for gen_dataset in  style_transfer_adain_with_clip_font2hand style_transfer_adain_font2hand;
do
    python3 train.py -g 1 --use_gen ../data_paths/gen/${gen_dataset} --pretraine &> experiment/train_etl_${gen_dataset}.log &
    wait
done
