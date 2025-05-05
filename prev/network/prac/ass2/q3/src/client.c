#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PORT 8080
#define WINDOW_SIZE 4
#define BUFFER_SIZE 1024
#define TOTAL_FRAMES 10

int main() {
  int sock = 0;
  struct sockaddr_in serv_addr;
  char buffer[BUFFER_SIZE] = {0};
  bool frame_ack[10] = {false};

  // Create socket file descriptor
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    printf("\n Socket creation error \n");
    return -1;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(PORT);

  // Convert IPv4 and IPv6 addresses from text to binary form
  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
    printf("\nInvalid address/ Address not supported \n");
    return -1;
  }

  // Connect to the server
  if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    printf("\nConnection Failed \n");
    return -1;
  }

  int base = 0;
  int next_seq_num = 0;

  while (base < TOTAL_FRAMES) {
    // Send frames within the window
    while (next_seq_num < base + WINDOW_SIZE && next_seq_num < TOTAL_FRAMES) {
      char frame[BUFFER_SIZE];
      sprintf(frame, "%d", next_seq_num);
      send(sock, frame, strlen(frame), 0);
      printf("Sent frame %d\n", next_seq_num);
      next_seq_num++;
    }

    // Receive ACKs
    fd_set readfds;
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;

    FD_ZERO(&readfds);
    FD_SET(sock, &readfds);

    int activity = select(sock + 1, &readfds, NULL, NULL, &tv);

    if (activity > 0) {
      int valread = read(sock, buffer, BUFFER_SIZE);
      if (valread > 0) {
        int acked_frame;
        sscanf(buffer, "ACK %d", &acked_frame);
        printf("Received ACK for frame %d\n", acked_frame);
        frame_ack[acked_frame] = true;

        if (acked_frame == base) {
          base++;
        }
      }
    } else {
      // Timeout occurred, retransmit unacked frames
      printf("Timeout occurred. Retransmitting unacked frames.\n");
      next_seq_num = base;
    }
  }

  close(sock);
  return 0;
}
