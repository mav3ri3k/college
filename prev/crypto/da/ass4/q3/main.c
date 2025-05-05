#include <stdio.h>
#include <stdint.h>


    const uint8_t fixed_mat[4][4] = {
        {0x02, 0x03, 0x01, 0x01},
        {0x01, 0x02, 0x03, 0x01},
        {0x01, 0x01, 0x02, 0x03},
        {0x03, 0x01, 0x01, 0x02}
    };


uint8_t mul(const uint8_t fixed_mat[4][4], uint8_t state_mat[4][4], int row, int col) {
  uint8_t res = 0;
  for (int i = 0; i < 4; i++) {
    // for first iter, xor might give wrong value
    // so just save it at first using or op
    if (i == 0) {
      res |= fixed_mat[row][i] * state_mat[i][col];
    } else {
      res ^= fixed_mat[row][i] * state_mat[i][col];
    }
  }

  return res;
}

void mix_columns(uint8_t state_mat[4][4]) {
    uint8_t temp[4][4] = {4};

    // iter over rows
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        temp[i][j] = mul(fixed_mat, state_mat, i, j); 
      }
    }

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        state_mat[i][j] = temp[i][j]; 
      }
    }
}

int main() {
    uint8_t state[4][4] = {
        {0x63, 0xEB, 0x9F, 0xA0},
        {0x2F, 0x93, 0x92, 0xC0},
        {0xAF, 0xC7, 0xAB, 0x30},
        {0xA2, 0x20, 0xCB, 0x2B}
    };

    printf("Original State:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%02x ", state[i][j]);
        }
        printf("\n");
    }

    mix_columns(state);

    printf("\nMix Columns:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%02x ", state[i][j]);
        }
        printf("\n");
    }

    return 0;
}
