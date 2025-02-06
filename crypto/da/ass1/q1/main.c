#include <ctype.h>
#include <stdio.h>
#include <string.h>

#define MAX_LEN 1000

void encrypt(char *text, int key) {
  for (int i = 0; text[i] != '\0'; i++) {
    if (isalpha(text[i])) {
      char base = isupper(text[i]) ? 'A' : 'a';
      text[i] = (text[i] - base + key) % 26 + base;
    }
  }
}

void decrypt(char *text, int key) {
  for (int i = 0; text[i] != '\0'; i++) {
    if (isalpha(text[i])) {
      char base = isupper(text[i]) ? 'A' : 'a';
      text[i] = (text[i] - base - key + 26) % 26 + base;
    }
  }
}

int main() {
  char text[MAX_LEN];
  int key;
  int op;

  printf("Enter text: ");
  fgets(text, MAX_LEN, stdin);
  text[strcspn(text, "\n")] = '\0';

  printf("Enter key value: ");
  scanf("%d", &key);

  printf("Choose operation: 1 for Encryption, 2 for Decryption: ");
  scanf("%d", &op);

  switch (op) {
  case 1:
    encrypt(text, key);
    printf("Encrypted text: %s\n", text);
    break;
  case 2:
    decrypt(text, key);
    printf("Decrypted text: %s\n", text);
    break;
  default:
    printf("Invalid choice!\n");
    break;
  }

  return 0;
}
