#!/bin/bash

print_stars() {
    for ((i=0; i<=3; i++)); do
        for ((j=1; j<=(3-i); j++)); do
            echo -n " "
        done
        for ((j=1; j<=(2*i+1); j++)); do
            echo -n "*"
        done
        echo
    done
}

print_numbers() {
    for ((i=3; i>=0; i--)); do
        for ((j=1; j<=(3-i); j++)); do
            echo -n " "
        done
        for ((j=1; j<=(2*i+1); j++)); do
            tmp=$((4-i))
            echo -n "$tmp"
        done
        echo
    done
}

print_stars
print_numbers
