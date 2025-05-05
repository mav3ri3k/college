#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT 8080
int main() {
  char tmp[255];
  fgets(tmp, 255, stdin);
  printf("%s\n", tmp);

  int sockfd, newsockfd;
  struct sockaddr_in saddr, caddr;
  char buffer[255];
  sockfd = socket(AF_INET, SOCK_DGRAM, 0);

  saddr.sin_family = AF_INET;
  saddr.sin_port = PORT;

  int res = bind(sockfd, (struct sockaddr *)&saddr, sizeof(saddr));
  listen(sockfd, 5);
  socklen_t len = sizeof(caddr);
  newsockfd = accept(sockfd, (struct sockaddr *)&caddr, &len);
  while (1) {
    res = read(newsockfd, buffer, 255);
    printf("Message: %s", buffer);
  }

  close(newsockfd);
  close(sockfd);
}
