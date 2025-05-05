#include <arpa/inet.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PORT 8080
#define WINDOW_SIZE 4
#define BUFFER_SIZE 1024
#define TOTAL_FRAMES 8

int main() {
  int skip_frame_1 = true;
  int server_fd, new_socket;
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);
  char buffer[BUFFER_SIZE] = {0};

  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(PORT);

  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }

  if (listen(server_fd, 3) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
  }

  printf("Server listening on port %d\n", PORT);

  if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                           (socklen_t *)&addrlen)) < 0) {
    perror("accept");
    exit(EXIT_FAILURE);
  }

  int base = 0;
  int next_seq_num = 0;
  int expected_seq_num = 0;
  bool received_frames[TOTAL_FRAMES] = {false};
  bool negative_frames[TOTAL_FRAMES] = {false};

  int prev = -1;
  while (1) {
    int valread = read(new_socket, buffer, BUFFER_SIZE);
    if (valread <= 0) {
      break;
    }
    for (int i = 0; i < BUFFER_SIZE; i++) {
      if (buffer[i] == 0) {
        break;
      } else {
        int current_frame = (int)(buffer[i] - '0');
        if (current_frame == 0 || current_frame % 5 == 4) {
          char nack[BUFFER_SIZE];
          sprintf(nack, "ACK %d", current_frame);
          send(new_socket, nack, strlen(nack), 0);
          printf("Sent ACK for frame %d\n", current_frame);
        }
        if (prev < 0) {
          prev = current_frame;
        } else {
          prev = current_frame - 1;
        }

        if (current_frame == 1 && skip_frame_1) {
          skip_frame_1 = false;
          continue;
        }

        received_frames[current_frame] = true;

        if (i == base) {
          base++;
          next_seq_num = base + WINDOW_SIZE;
        }
        printf("Received Frames: ");
        for (int i = 0; i < TOTAL_FRAMES; i++) {
          if (received_frames[i]) {
            printf(" %d ", i);
          }
        }
        printf("\n");

        for (int j = base; j <= current_frame; j++) {
          if (received_frames[j] == false && negative_frames[j] == false) {
            char nack[BUFFER_SIZE];
            sprintf(nack, "ACK %d", -j);
            send(new_socket, nack, strlen(nack), 0);
            printf("Sent NACK for frame %d\n", -j);
            negative_frames[j] = true;
            break;
          }
        }
      }
    }
  }

  close(new_socket);
  close(server_fd);
  return 0;
}
