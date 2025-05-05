#include <memory.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT 8080

int main() {
  int sock_fd, newsock_fd;
  char buffer[255] = {'\0'};

  struct sockaddr_in server_addr, client_addr;

  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    perror("Server socket could not be created");
    exit(0);
  }

  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port = htons(PORT);

  int status =
      bind(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));

  if (status < 0) {
    perror("Failed to bind to socket");
    exit(0);
  }

  status = listen(sock_fd, 5);
  unsigned int caddrlen = sizeof(client_addr);
  newsock_fd = accept(sock_fd, (struct sockaddr *)&client_addr, &caddrlen);

  if (newsock_fd < 0) {
    perror("Error on accept");
    exit(0);
  }

  while (true) {
    status = read(newsock_fd, buffer, sizeof(buffer));
    if (status < 0) {
      perror("Could not read");
    }
    printf("Client says: %s", buffer);

    status = strncmp("bye", buffer, 3);

    if (status == 0) {
      break;
    }

    memset(buffer, '\0', sizeof(buffer));

    fgets(buffer, sizeof(buffer), stdin);

    status = write(newsock_fd, buffer, strlen(buffer));
    if (status < 0) {
      printf("Error on writing");
    }
  }

  close(newsock_fd);
  close(sock_fd);
}
