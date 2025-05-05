s:
     zig cc  ./src/server.c -std=c23 -o ./bin/server
sr: s
    ./bin/server
c:
     zig cc  ./src/client.c -std=c23 -o ./bin/client
cr: c
    ./bin/client

