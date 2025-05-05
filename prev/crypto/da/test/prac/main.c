#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int e_gcd(int a, int b) {
  while (b != 0) {
    int tmp = b;
    b = a % b;
    a = tmp;
  }

  return a;
}

int mod_inverse(int a, int p) {
  for (int i = 1; i < 500; i++) {
    if ((i * a) % p == 1) {
      return i;
    }
  }

  fprintf(stderr, "Mod Inverse could not be found!\n");
  exit(0);
}

int main() {
  printf("Ans: %d\n", mod_inverse(2, 17));
  int **table;
  table = malloc(10 * sizeof(*table));
  for (int i = 0; i < 10; i++) {
    table[i] = malloc(10 * sizeof(table[0]));
  }


  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      table[i][j] = i;
    }
  }

  
  // for (int i = 0; i < 10; i++) {
  //   for (int j = 0; j < 10; j++) {
  //     printf("%d ",  table[i][j] = i);
  //   }
  //   puts("");
  // }
  char input[100];
  memset(input, '\0', 100);
  fgets(input, 100, stdin);
  printf("len: %lu\n", strlen(input));
  int x, y;
  x = 0;
  y = 0;
  printf("len: %d\n", e_gcd(8, 2));
  return 0;
}
