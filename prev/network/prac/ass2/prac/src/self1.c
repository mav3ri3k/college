#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT 8080

int main() {
  char buffer[255] = {'\0'};
  int sock_fd, newsock_fd, status;

  sock_fd = socket(AF_INET, SOCK_STREAM, 0);

  struct sockaddr_in server_addr, client_addr;

  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port = PORT;

  status = bind(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));

  status = listen(sock_fd, 5);
  unsigned int caddrlen = sizeof(client_addr);

  newsock_fd = accept(sock_fd, (struct sockaddr *)&client_addr, &caddrlen);

  while (true) {
    status = read(newsock_fd, buffer, sizeof(buffer));
    printf("Buffer: %s", buffer);
    fgets(buffer, sizeof(buffer), stdin);
    status = write(newsock_fd, buffer, strlen(buffer));
  }

  close(newsock_fd);
  close(sock_fd);
}
