l:
     zig cc  ./src/lexer.c -std=c23 -o ./bin/lexer
lr: l
    ./bin/lexer
t:
     zig cc  ./src/table.c -std=c23 -o ./bin/table
tr: t
    ./bin/table

