#!/bin/bash

lex main.l
yacc -d main.y
gcc lex.yy.c y.tab.c -o main
./main
