#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PORT 8080
#define WINDOW_SIZE 4
#define BUFFER_SIZE 1024

int main() {
  int server_fd, new_socket;
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);
  char buffer[BUFFER_SIZE] = {0};

  // Create socket file descriptor
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(PORT);

  // Bind the socket to the network address and port
  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }

  // Listen for incoming connections
  if (listen(server_fd, 3) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
  }

  printf("Server listening on port %d\n", PORT);

  // Accept incoming connection
  if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
                           (socklen_t *)&addrlen)) < 0) {
    perror("accept");
    exit(EXIT_FAILURE);
  }

  int base = 0;
  int next_seq_num = 0;
  int expected_seq_num = 0;
  bool received_frames[10] = {false};

  while (1) {
    // Receive data from client
    int valread = read(new_socket, buffer, BUFFER_SIZE);
    if (valread <= 0) {
      break;
    }

    for (int i = 0; i < BUFFER_SIZE; i++) {
      if (buffer[i] == 0) {
        break;
      } else {
        int current_frame = (int)(buffer[i] - '0');
        received_frames[current_frame] = true;
        char ack[BUFFER_SIZE];
        sprintf(ack, "ACK %d", current_frame);
        send(new_socket, ack, strlen(ack), 0);
        printf("Sent ACK for frame %d\n", current_frame);

        // Move the window if necessary
        if (i == base) {
          base++;
          next_seq_num = base + WINDOW_SIZE;
        }
      }
    }

    printf("Received Frames: ");
    for (int i = 0; i < 10; i++) {
      if (received_frames[i]) {
        printf(" %d ", i);
      }
    }
    printf("\n");
  }

  close(new_socket);
  close(server_fd);
  return 0;
}
