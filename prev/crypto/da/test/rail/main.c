#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct table_index {
  int row;
  int column;
  int down;
};

void inc_index(struct table_index* index, int height) {
  index -> column += 1;

  if (index->down) {
    index -> row += 1;
  } else {
    index -> row -= 1;
  }

  if (index->row == height-1 || index->row == 0) {
    index -> down = !index->down;
  }
}

void set_char_at_index(char ch, struct table_index index, char** table) {
  table[index.row][index.column] = ch;
}

void encrypt(char* input, char** table, int height, int len) {
  struct table_index index = {0, 0, 1};

  for (int i = 0; i < len; i++) {
    table[index.row][index.column] = input[i];
    printf("%d\n", i);
    inc_index(&index, height);
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < len; j++) {
      printf("%c ", table[i][j]);
    }
    printf("\n");
  }

  puts("table done");
  int input_index = 0;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < len; j++) {
      char ch = table[i][j];
      if (ch != '\0') {
        printf("%c\n", ch);
        input[input_index] = ch;
      input_index += 1;
      }
    }
  }
  puts(input);
}

int main() {
  char input[100];
  memset(input, '\0', 100);
  fgets(input, 100, stdin);
  int len = strlen(input) - 1;
  printf("Len: %d\n", len);

  int height;
  scanf("%d", &height);

  char **table;
  table = malloc(height * sizeof(*table));
  for (int i = 0; i < height; i++) {
    table[i] = malloc(len * sizeof(table[0]));
  }
  puts("table cratead");

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < len; j++) {
      table[i][j] = '\0';
    }
  }

  puts("table cratead");
  encrypt(input, table, height, len);

  puts("table cratead");
  puts(input);

  free(table);
  return 0;
} 
