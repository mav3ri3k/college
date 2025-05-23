#include <stdio.h>
#include <stdint.h>

int p[64] = {58, 50, 42, 34, 26, 18, 10, 2, 60, 52, 44, 36, 28, 20, 12, 4, 62, 54, 46, 38, 30, 22, 14, 6, 64, 56, 48, 40, 32, 24, 16, 8, 57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35, 27, 19, 11, 3, 61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31, 23, 15, 7};

uint64_t init_perm(uint64_t input) {
  uint64_t out = 0;
  for (int i = 0; i < 64; i++) {
    uint8_t num = 0;
    uint64_t t_in = input;
    for (int j = 0; j < 64 - p[i]; j++) {
      t_in = t_in >> 1;
    }

    t_in = t_in & 1;

    out = out << 1;
    out = out | t_in;
  }

  return out;
}

int main() {
  uint64_t input;
  scanf("%llx", &input);
  printf("%llx\n", input);

  printf("%llx\n", init_perm(input));
  return 0;
}
