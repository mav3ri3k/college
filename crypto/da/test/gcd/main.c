#include <stdio.h>
#include <string.h>

int gcd(int a, int b) {
  while(b != 0) {
    int tmp = b;
    b = a % b;
    a = tmp;
  }

  return a;
}

int main() {
  printf("%d\n", gcd(10, 5));
  printf("%d\n", gcd(71, 5));
  printf("%d\n", gcd(2, 4));
  return 0;
}
