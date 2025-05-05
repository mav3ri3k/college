#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define PORT 8080

int main(int argc, char const *argv[]) {
  int server_fd, new_socket;
  struct sockaddr address;
  int opt = 1;
  int addrlen = sizeof(address);

  // Creating socket file descriptor
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  // Forcefully attaching socket to the port 8080
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                 sizeof(opt))) {
    perror("setsockopt");
    exit(EXIT_FAILURE);
  }
  address.sa_family = AF_INET;
  // No need to set address.sin_addr.s_addr and address.sin_port since we're
  // using the generic sockaddr struct

  // Binding the socket to the address and port
  if (bind(server_fd, &address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }

  // Listening for incoming connections
  if (listen(server_fd, 3) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
  }

  printf("Server is listening on port %d\n", PORT);

  // Accept an incoming connection
  if ((new_socket = accept(server_fd, &address, (socklen_t *)&addrlen)) < 0) {
    perror("accept");
    exit(EXIT_FAILURE);
  }

  // Handle the connection (you can add your own logic here)
  char buffer[1024] = {0};
  int valread = read(new_socket, buffer, 1024);
  printf("%s\n", buffer);
  send(new_socket, "Hello from the server!", 21, 0);

  // Clean up
  close(new_socket);
  close(server_fd);
  return 0;
}
