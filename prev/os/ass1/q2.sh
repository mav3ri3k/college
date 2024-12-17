#!/bin/bash

sum=0
echo "Number of numbers to add:"
read n

echo "Enter the numbers:"

for ((i=1; i<=n; i++))
do
  read num
  sum=$((sum + num))
done

echo "The sum is: $sum"
