#include <ctype.h>
#include <stdio.h>
#include <string.h>

#define MAX_LEN 1000

void encrypt(char *text, char *key) {
  int textLen = strlen(text);
  int keyLen = strlen(key);
  char encryptedText[MAX_LEN];

  for (int i = 0, j = 0; i < textLen; i++) {
    if (isalpha(text[i])) {
      char base = isupper(text[i]) ? 'A' : 'a';
      char keyBase = isupper(key[j % keyLen]) ? 'A' : 'a';
      encryptedText[i] =
          (text[i] - base + (key[j % keyLen] - keyBase)) % 26 + base;
      j++;
    } else {
      encryptedText[i] = text[i];
    }
  }
  encryptedText[textLen] = '\0';
  printf("Encrypted Text: %s\n", encryptedText);
}

void decrypt(char *text, char *key) {
  int textLen = strlen(text);
  int keyLen = strlen(key);
  char decryptedText[MAX_LEN];

  for (int i = 0, j = 0; i < textLen; i++) {
    if (isalpha(text[i])) {
      char base = isupper(text[i]) ? 'A' : 'a';
      char keyBase = isupper(key[j % keyLen]) ? 'A' : 'a';
      decryptedText[i] =
          (text[i] - base - (key[j % keyLen] - keyBase) + 26) % 26 + base;
      j++;
    } else {
      decryptedText[i] = text[i];
    }
  }
  decryptedText[textLen] = '\0';
  printf("Decrypted Text: %s\n", decryptedText);
}

int main() {
  char text[MAX_LEN], key[MAX_LEN];
  int choice;

  printf("Enter text: ");
  fgets(text, MAX_LEN, stdin);
  text[strcspn(text, "\n")] = '\0';

  printf("Enter key: ");
  fgets(key, MAX_LEN, stdin);
  key[strcspn(key, "\n")] = '\0';

  printf("Choose operation: 1 for Encryption, 2 for Decryption: ");
  scanf("%d", &choice);
  getchar(); // Consume newline

  if (choice == 1) {
    encrypt(text, key);
  } else if (choice == 2) {
    decrypt(text, key);
  } else {
    printf("Invalid choice!\n");
  }

  return 0;
}
