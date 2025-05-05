#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT 8080
int main() {
  int sock;
  struct sockaddr_in saddr;
  char buffer[255];
  fgets(buffer, 255, stdin);
  sock = socket(AF_INET, SOCK_DGRAM, 0);

  saddr.sin_family = AF_INET;
  saddr.sin_addr.s_addr = INADDR_ANY;
  saddr.sin_port = htons(PORT);

  int res = connect(sock, (struct sockaddr *)&saddr, sizeof(saddr));
  if (res < 0) {
    puts("error");
    close(sock);
    exit(1);
  }

  write(sock, buffer, 100);

  close(sock);
}
