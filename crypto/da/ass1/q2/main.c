#include <ctype.h>
#include <stdio.h>
#include <string.h>

#define SIZE 5
#define MAX_TEXT 100

char keySquare[SIZE][SIZE];

void generateKeySquare(const char *key) {
  int map[26] = {0};
  int x = 0, y = 0;
  char processedKey[26] = "";
  int index = 0;

  for (int i = 0; key[i] != '\0'; i++) {
    char ch = toupper(key[i]);
    if (ch == 'J')
      ch = 'I';
    if (!map[ch - 'A'] && isalpha(ch)) {
      map[ch - 'A'] = 1;
      processedKey[index++] = ch;
    }
  }

  for (char ch = 'A'; ch <= 'Z'; ch++) {
    if (ch == 'J')
      continue;
    if (!map[ch - 'A']) {
      processedKey[index++] = ch;
    }
  }

  index = 0;
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      keySquare[i][j] = processedKey[index++];
    }
  }
}

void findPosition(char ch, int *row, int *col) {
  if (ch == 'J')
    ch = 'I';
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      if (keySquare[i][j] == ch) {
        *row = i;
        *col = j;
        return;
      }
    }
  }
}

void prepareText(char *text) {
  int len = strlen(text);
  for (int i = 0; i < len; i++) {
    text[i] = toupper(text[i]);
    if (text[i] == 'J')
      text[i] = 'I';
  }

  char newText[MAX_TEXT];
  int newIndex = 0;

  for (int i = 0; i < len; i++) {
    if (!isalpha(text[i]))
      continue;
    newText[newIndex++] = text[i];
    if (i + 1 < len && text[i] == text[i + 1]) {
      newText[newIndex++] = 'X';
    }
  }

  if (newIndex % 2 != 0) {
    newText[newIndex++] = 'X';
  }
  newText[newIndex] = '\0';
  strcpy(text, newText);
}

void playfairCipher(char *text, int encrypt) {
  for (int i = 0; i < strlen(text); i += 2) {
    int r1, c1, r2, c2;
    findPosition(text[i], &r1, &c1);
    findPosition(text[i + 1], &r2, &c2);

    if (r1 == r2) {
      text[i] = keySquare[r1][(c1 + encrypt + SIZE) % SIZE];
      text[i + 1] = keySquare[r2][(c2 + encrypt + SIZE) % SIZE];
    } else if (c1 == c2) {
      text[i] = keySquare[(r1 + encrypt + SIZE) % SIZE][c1];
      text[i + 1] = keySquare[(r2 + encrypt + SIZE) % SIZE][c2];
    } else {
      text[i] = keySquare[r1][c2];
      text[i + 1] = keySquare[r2][c1];
    }

    text[i] = tolower(text[i]);
    text[i + 1] = tolower(text[i + 1]);
  }
}

int main() {
  char key[MAX_TEXT], text[MAX_TEXT];
  int op;

  printf("Enter key: ");
  fgets(key, MAX_TEXT, stdin);
  key[strcspn(key, "\n")] = '\0';

  generateKeySquare(key);

  printf("Enter text: ");
  fgets(text, MAX_TEXT, stdin);
  text[strcspn(text, "\n")] = '\0';

  prepareText(text);

  printf("Choose operation (1 for Encryption, 2 for Decryption): ");
  scanf("%d", &op);

  switch (op) {
  case 1:
    playfairCipher(text, 1);
    printf("Encrypted text: %s\n", text);
    break;
  case 2:
    playfairCipher(text, -1);
    printf("Decrypted text: %s\n", text);
    break;
  default:
    printf("Invalid choice!\n");
    break;
  }

  return 0;
}
