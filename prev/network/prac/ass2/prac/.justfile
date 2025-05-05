s:
     cc  ./src/server.c -std=c2x -o ./bin/server
sr: s
    ./bin/server
c:
     cc  ./src/client.c -std=c2x -o ./bin/client
cr: c
    ./bin/client

