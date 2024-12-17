#!/bin/bash

declare -a array

echo "Enter the number of elements:"
read n

echo "Enter the elements:"

for ((i=0; i<n; i++))
do
  read num
  array[i]=$num
done

sum=0

for i in "${array[@]}"
do
  sum=$((sum + i))
done

echo "The sum of array elements is: $sum"
