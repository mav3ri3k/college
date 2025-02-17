#include <stdio.h>
#include <stdint.h>

// left rotate row once
void left_rotate(uint8_t *row) {
  uint8_t first = row[0];

  for (int i = 0; i < 3; i++) {
    row[i] = row[i + 1];
  }

  row[3] = first;
}

void shift_rows(uint8_t state[4][4]) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < i; j++) {
      left_rotate(state[i]);
    }
  }
}

int main() {
    uint8_t state[4][4] = {
        {0xd4, 0xe0, 0xb8, 0x1e},
        {0xbf, 0xb4, 0x52, 0xa0},
        {0xae, 0xb8, 0x41, 0x11},
        {0xf7, 0xdc, 0x82, 0x9a}
    };

    printf("Original State:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%02x ", state[i][j]);
        }
        printf("\n");
    }

    shift_rows(state);

    printf("\nShifted State:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%02x ", state[i][j]);
        }
        printf("\n");
    }

    return 0;
}
