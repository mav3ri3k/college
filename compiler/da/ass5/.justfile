default: build run

build:
    lex alphabets.l
    zig cc lex.yy.c -o alphabets;

run:
    ./alphabets
