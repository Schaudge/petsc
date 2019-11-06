#!/bin/bash
total=64
for (( i=1; i<=$total; i++ ))
do
  tend=`echo 0.25*$i/$total | bc -l`
  echo $tend
  ./ex1adj -pc_type lu  -tend $tend
done

./ex1fwd -pc_type lu  -tend 0.25
