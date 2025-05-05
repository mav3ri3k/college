#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

#define PORT 8080
#define WINDOW_SIZE 4
#define BUFFER_SIZE 10

int main() {
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    perror("Connection could not be established\n");
    exit(1);
  }

  struct sockaddr address;
  address.sa_family = AF_INET;
  address.sa_data = "127.0.0.1";

  struct sockaddr_in addr_in;
  addr_in.sa_family_t = AF_INET;
  addr_in.in_port_t = PORT;
  addr_in.sin_addr = address;

  int res = bind(socket_fd, &address, sizeof(address));
  if (res != 0) {
    perror("Socket could not be bound\n");
    exit(2);
  }
}
