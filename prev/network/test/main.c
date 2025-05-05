#include <stdio.h>
#include <string.h>

int main() {
  char str1[10];
  char str2[10];
  scanf("%s", str1);
  printf("String1: %s\n", str1);
  strcpy(str2, str1);
  memset(str2 + 2, '\0', 8 * sizeof(char));
  printf("String1: %s\n", str2);
}
