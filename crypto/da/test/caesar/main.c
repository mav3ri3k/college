#include <stdio.h>
#include <string.h>

char move(char input, int move) {
  int normalised_encoding = input - 'a';

  int move_encoding = normalised_encoding + move;
  while (move_encoding < 0 || move_encoding >= 26) {
    if (move < 0) {
      move_encoding += 26;
    } else {
      move_encoding -= 26;
    }
  }

  return (char) move_encoding + 'a';
}

void encrypt(char *text, int key) {
  int l = strlen(text);

  for (int i = 0; i < l; i++) {
    text[i] = move(text[i], key);
  }
}

void decrypt(char *text, int key) {
  int l = strlen(text);

  for (int i = 0; i < l; i++) {
    text[i] = move(text[i], -key);
  }
}

int main() {

  char input[100];
  memset(input, '\0', 100);
  fgets(input, 100, stdin);
  puts(input);

  int key;
  scanf("%d", &key);

  encrypt(input, key);
  puts("Encrypted Text:");
  puts(input);
  decrypt(input, key);
  puts("Decrypted Text:");
  puts(input);
  
  return 0;
}
