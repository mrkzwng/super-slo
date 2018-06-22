#! /bin/bash

for N in {1..9}; do echo $N; cat $( find $1 -name \*_frame$N.txt | sort -u )  > ./data_list/frame$N.txt; done