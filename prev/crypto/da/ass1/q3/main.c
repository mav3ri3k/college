#include <ctype.h>
#include <stdio.h>
#include <string.h>

#define MAX_LEN 1000

void encrypt(char *text, int key) {
  int len = strlen(text);
  char rail[key][len];

  for (int i = 0; i < key; i++) {
    for (int j = 0; j < len; j++) {
      rail[i][j] = '\n';
    }
  }

  int row = 0, dir_down = 0;
  for (int i = 0; i < len; i++) {
    if (row == 0 || row == key - 1) {
      dir_down = !dir_down;
    }
    rail[row][i] = text[i];
    row += (dir_down) ? 1 : -1;
  }

  printf("Encrypted Text: ");
  for (int i = 0; i < key; i++) {
    for (int j = 0; j < len; j++) {
      if (rail[i][j] != '\n') {
        printf("%c", rail[i][j]);
      }
    }
  }
  printf("\n");
}

void decrypt(char *cipher, int key) {
  int len = strlen(cipher);
  char rail[key][len];

  for (int i = 0; i < key; i++) {
    for (int j = 0; j < len; j++) {
      rail[i][j] = '\n';
    }
  }

  int row = 0;
  int dir_down = 0;
  for (int i = 0; i < len; i++) {
    if (row == 0 || row == key - 1) {
      dir_down = !dir_down;
    }
    rail[row][i] = '*';
    row += (dir_down) ? 1 : -1;
  }

  int index = 0;
  for (int i = 0; i < key; i++) {
    for (int j = 0; j < len; j++) {
      if (rail[i][j] == '*' && index < len) {
        rail[i][j] = cipher[index++];
      }
    }
  }

  row = 0, dir_down = 0;
  printf("Decrypted Text: ");
  for (int i = 0; i < len; i++) {
    if (row == 0 || row == key - 1) {
      dir_down = !dir_down;
    }
    printf("%c", rail[row][i]);
    row += (dir_down) ? 1 : -1;
  }
  printf("\n");
}

int main() {
  char text[MAX_LEN];
  int key, choice;

  printf("Enter text: ");
  fgets(text, MAX_LEN, stdin);
  text[strcspn(text, "\n")] = '\0';

  printf("Enter key (number of rails): ");
  scanf("%d", &key);

  printf("Choose operation: 1 for Encryption, 2 for Decryption: ");
  scanf("%d", &choice);
  getchar();

  if (choice == 1) {
    encrypt(text, key);
  } else if (choice == 2) {
    decrypt(text, key);
  } else {
    printf("Invalid choice!\n");
  }

  return 0;
}
