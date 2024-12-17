#!/bin/bash

is_palindrome() {
  num=$1
  reversed=$(echo $num | rev)

  if [ "$num" -eq "$reversed" ]; then
    echo "The number $num is a palindrome"
  else
    echo "The number $num is not a palindrome"
  fi
}

read -p "Enter a number: " num

is_palindrome $num
