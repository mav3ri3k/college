#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

void check(int status, char emessage[], char smessage[]) {
  if (status < 0) {
    perror(emessage);
    exit(0);
  }
  puts(smessage);
}

int main() {
  int sock_fd, status;
  char buffer[255] = {'\0'};

  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  check(sock_fd, "Socket could not be created", "Socket created");

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(8080);
  server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

  status =
      connect(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr));
  check(status, "Could not connect", "Connection successful");

  while (true) {
    printf("Message: ");
    fgets(buffer, sizeof(buffer), stdin);

    status = write(sock_fd, buffer, sizeof(buffer));
    check(status, "Could not write", "Write success");

    status = strncmp("bye", buffer, 3);
    if (status == 0) {
      break;
    }

    memset(buffer, '\0', sizeof(buffer));

    status = read(sock_fd, buffer, sizeof(buffer));
    check(status, "Could not read", "Read successful");

    printf("Server says: %s", buffer);

    memset(buffer, '\0', sizeof(buffer));
  }

  close(sock_fd);

  return (0);
}
