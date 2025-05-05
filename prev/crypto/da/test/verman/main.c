#include <stdio.h>
#include <string.h>

char move_char(char input, int move) {
  int normalised_char = input - 'a';
  int move_char = normalised_char + move;

  while (move_char < 0 || move_char >= 26) {
    if (move > 0) {
      move_char -= 26;
    } else {
      move_char += 26;
    }
  }

  return (char)(move_char + 'a');
}

int n_key(char *key, int index) { return key[index] - 'a'; }

void encrypt(char *input, char *key) {
  int l = strlen(input);
  int k_l = strlen(key);
  int key_index = 0;
  for (int i = 0; i < l; i++) {
    input[i] = move_char(input[i], n_key(key, key_index));

    key_index += 1;
    if (key_index > k_l) {
      key_index = 0;
    }
  }
}

void decrypt(char *input, char *key) {
  int l = strlen(input);
  int k_l = strlen(key);
  int key_index = 0;
  for (int i = 0; i < l; i++) {
    input[i] = move_char(input[i], -n_key(key, key_index));

    key_index += 1;
    if (key_index > k_l) {
      key_index = 0;
    }
  }
}
int main() {
  char input[100];
  memset(input, '\0', 100);
  fgets(input, 100, stdin);

  char key[100];
  memset(key, '\0', 100);
  fgets(key, 100, stdin);

  puts("Input & key: ");
  puts(input);
  puts(key);

  encrypt(input, key);
  puts("Encrpted Text: ");
  puts(input);

  decrypt(input, key);
  puts("Decrypted Text: ");
  puts(input);
  return 0;
}
