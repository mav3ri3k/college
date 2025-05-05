#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PORT 8080
#define WINDOW_SIZE 4
#define BUFFER_SIZE 1024
#define TOTAL_FRAMES 8

int main() {
  bool skip_1 = true;
  int sock = 0;
  struct sockaddr_in serv_addr;
  char buffer[BUFFER_SIZE] = {0};
  bool frame_ack[10] = {false};

  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    printf("\n Socket creation error \n");
    return -1;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(PORT);

  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
    printf("\nInvalid address/ Address not supported \n");
    return -1;
  }

  if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    printf("\nConnection Failed \n");
    return -1;
  }

  int base = 0;
  int next_seq_num = 0;

  while (base < TOTAL_FRAMES) {
    // Send frames within the window
    while (next_seq_num < base + WINDOW_SIZE && next_seq_num < TOTAL_FRAMES) {
      if (skip_1 && next_seq_num == 1) {
        skip_1 = false;
        next_seq_num += 1;
        continue;
      }
      char frame[BUFFER_SIZE];
      sprintf(frame, "%d", next_seq_num);
      send(sock, frame, strlen(frame), 0);
      printf("Sent frame %d\n", next_seq_num);
      next_seq_num++;
      if (next_seq_num == 1) {
        break;
      }
    }
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
        if (acked_frame < 0) {
          acked_frame = (-acked_frame);
          printf("Received negative acknowledge: %d\n", acked_frame);

          char frame[BUFFER_SIZE];
          sprintf(frame, "%d", acked_frame);
          send(sock, frame, strlen(frame), 0);
          printf("Sent frame %d\n", acked_frame);
          continue;
        }

        if (acked_frame > TOTAL_FRAMES) {
          acked_frame /= 10;
        }

        printf("Received ACK for frame %d\n", acked_frame);
        frame_ack[acked_frame] = true;

        if (acked_frame == base) {
          base++;
        }
      }
    } else {
      if (skip_1) {
        skip_1 = false;
        continue;
      }
      printf("Timeout occured\n");
      next_seq_num = base;
    }
  }

  close(sock);
  return 0;
}
