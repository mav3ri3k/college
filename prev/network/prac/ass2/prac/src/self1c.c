#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT 8080

int main() {
  printf("Starting client!");
  int sock_fd, status;
  char buffer[255] = {'\0'};
  char lb_addr[] = "127.0.0.1";

  struct sockaddr_in server_addr;
  struct hostent *server;

  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server == NULL) {
    perror("Server is off");
  }

  printf("Getting server info");
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = inet_addr(lb_addr);
  server_addr.sin_port = htons(PORT);

  status =
      connect(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
  if (status < 0) {
    perror("Could not connect\n");
  }

  while (true) {
    printf("Input: \n");
    fgets(buffer, sizeof(buffer), stdin);
    status = write(sock_fd, buffer, strlen(buffer));
    if (status < 0) {
      perror("Could not write");
    }

    status = strncmp("bye", buffer, 3);
    if (status == 0) {
      break;
    }

    status = read(sock_fd, buffer, sizeof(buffer));
    if (status < 0) {
      perror("Error on read");
    }

    printf("Server: %s", buffer);
  }

  close(sock_fd);
}
