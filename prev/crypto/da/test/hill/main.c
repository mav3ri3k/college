#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_input(char *input, char **table, int len) {
  for (int i = 0; i < len; i++) {
    for (int j = 0; j < len; j++) {
      table[i][j] = (rand() % 26);
    }
  }

  for (int i = 0; i < len; i++) {
    input[i] -= 'a';
  }
}

void encrypt(char *input, char *output, char **table, int len) {
  for (int i = 0; i < len; i++) {
    int tmp = 0;
    for (int j = 0; j < len; j++) {
        tmp += table[i][j]*input[j];    
    }

    output[i] = (tmp % 26) + 'a';
  }
}

int main() {
  char input[100];
  memset(input, '\0', 100);
  fgets(input, 100, stdin);

  int len = strlen(input) - 1;

  char *output = malloc(len * sizeof(char));
  memset(output, 'a' - 'a', len);

  char **table = malloc(len * sizeof(*table));
  for (int i = 0; i < len; i++) {
    table[i] = malloc(len * sizeof(table[0]));
  }

  for (int i = 0; i < len; i++) {
    for (int j = 0; j < len; j++) {
      table[i][j] = 0;
    }
  }

  init_input(input, table, len);
  encrypt(input, output, table, len);
  printf("Encrypted Text: %s\n", output);

  free(table);
  free(output);
  return 0;
}
