#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum type {
  Identifier,
  Delimiter,
  Operator,
  Keyword,
};

struct item {
  char name[3];
  enum type type;
  int id;
};

char keywords[5][4] = {"int", "else", "for", "fn", "if"};
char operators[2][2] = {"=", "+"};
char delimeters[3][2] = {",", "//", ";"};

int isize = 14;
char *input;
// char input[] = "int a, b = 10;";

char *read_file(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (fp == NULL) {
    perror("Error opening file");
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  rewind(fp);

  char *buffer = (char *)malloc(size + 1);
  if (buffer == NULL) {
    perror("Memory allocation failed");
    fclose(fp);
    return NULL;
  }

  if (fread(buffer, 1, size, fp) != size) {
    perror("Error reading file");
    free(buffer);
    fclose(fp);
    return NULL;
  }
  isize = size;
  buffer[size] = '\0'; // Null-terminate the string
  fclose(fp);
  return buffer;
}

bool is_operator(char *operator) {
  for (int i = 0; i < 2; i++) {
    int result = strcmp(operator, operators[i]);
    if (result == 0) {
      return true;
    }
  }
  return false;
}

bool is_keyword(char *keyword) {
  for (int i = 0; i < 5; i++) {
    int result = strcmp(keyword, keywords[i]);
    if (result == 0) {
      return true;
    }
  }
  return false;
}

bool is_delimiter(char *delimeter) {
  for (int i = 0; i < 3; i++) {
    int result = strcmp(delimeter, delimeters[i]);
    if (result == 0) {
      return true;
    }
  }
  return false;
}

// print char in given range for array: input
void print_chars(int start, int end) {
  for (int i = start; i < end; i++) {
    printf("%c", input[i]);
  }
  printf("\n");
}

// remove given index from array: input
void r_index(int index) {
  if (index <= 0) {
    return;
  }
  for (int i = index; i < isize; i++) {
    input[i] = input[i + 1];
  }
}

// remove whitespace from global scopce array: input
// remove \n from global scopce array: input
// returns newsize
void r_wn() {
  for (int i = 0; i < isize; i++) {
    if (input[i] == ' ') {
      r_index(i);
      isize -= 1;
    } else if (input[i] == '\n') {
      r_index(i);
      isize -= 1;
    }
  }
}

void lexer() {
  int token_found = 0;
  int index_prev = 0;

  for (int j = 0; j < isize; j++) {
    int pointer = 0;
    char token[4] = {'\0'};

    // assume, max token size = 3
    for (int i = 0; i < 3; i++) {

      token[pointer] = input[j + i];
      pointer += 1;

      if (is_keyword(token)) {
        printf("  Keyword: ");
        print_chars(j, j + pointer);
        token_found = 1;
        break;
      } else if (is_delimiter(token)) {
        printf("  Delimiter: ");
        print_chars(j, j + pointer);
        token_found = 1;
        break;
      } else if (is_operator(token)) {
        printf("  Operator: ");
        print_chars(j, j + pointer);
        token_found = 1;
        break;
      }
    }

    if (token_found == 1) {
      token_found = 0;

      if (index_prev != 0) {
        // identifier
        printf("  Identifier: ");
        print_chars(index_prev, j);
      }
      index_prev = j + pointer;

      j += pointer;
    }

    pointer = 0;
    // reset token
    for (int m = 0; m < 3; m++) {
      token[m] = '\0';
    }
  }
}

int main() {
  input = read_file("input.txt");
  printf("Input:\n  %s\n", input);
  r_wn();
  printf("Sanitized input:\n  %s\n\n", input);
  printf("Tokens: \n");
  lexer();
  return 0;
}
