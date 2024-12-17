default: build run

build:
    lex main.l
    zig cc lex.yy.c -o main;

run:
    ./main
