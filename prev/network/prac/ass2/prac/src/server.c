#include <arpa/inet.h>
#include <memory.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT 8080

void check(int status, char message[]) {
  if (status < 0) {
    perror(message);
    exit(0);
  }
}

int main() {
  int sock_fd, newsock_fd, status;
  char buffer[255] = {'\0'};

  sock_fd = socket(AF_INET, SOCK_STREAM, 0);

  check(sock_fd, "Server: Could not connect to socket");

  struct sockaddr_in server_addr, client_addr;

  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(PORT);
  server_addr.sin_addr.s_addr = INADDR_ANY;

  status = bind(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
  check(status, "Serer: Could not bind");
  puts("Server: Bind successfull");

  status = listen(sock_fd, 5);
  check(status, "Server: Could not listen");
  puts("Server: Listening..");

  unsigned int c_addr_len = sizeof(client_addr);

  newsock_fd = accept(sock_fd, (struct sockaddr *)&client_addr, &c_addr_len);
  check(newsock_fd, "Server: Client connection failed");
  puts("New client connected");

  while (1) {
    status = read(newsock_fd, buffer, sizeof(buffer));
    check(status, "Server: Read unsuccessful");

    printf("Server: Client says: %s", buffer);
    status = strncmp("bye", buffer, 3);
    if (status == 0) {
      break;
    }
    memset(buffer, '\0', sizeof(buffer));

    printf("Server: Message: ");
    fgets(buffer, sizeof(buffer), stdin);

    status = write(newsock_fd, buffer, strlen(buffer));
  }

  puts("Server: Hala lui baby!");

  close(newsock_fd);
  close(sock_fd);

  return (0);
}
