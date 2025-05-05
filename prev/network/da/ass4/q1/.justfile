sr: sb
    ./bin/server
sb:
    zig cc ./src/server.c -o ./bin/server --std=c23
cr: cb
    ./bin/client
cb:
    zig cc ./src/client.c -o ./bin/client --std=c23
    

