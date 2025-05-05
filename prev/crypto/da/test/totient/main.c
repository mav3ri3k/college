#include <stdio.h>

int gcd(int a, int b) {
  while (b != 0) {
    int tmp = b;
    b = a % b;
    a = tmp;
  }

  return a;
}

int is_prime(int n) {
  for (int i = 2; i < n; i++) {
    if (gcd(i ,n) != 1) {
      return 0;
    }
  }

  return 1;
}

int totient(int n) {
  int top = -1;
  int p_factors[100] = {0};
  for (int i = 2; i <= n; i++) {
    if (is_prime(i) && gcd(i, n) != 1) {
      top += 1;
      p_factors[top] = i;
    }
  }

  int tot = 1;
  for (int i = 0; i <= top; i++) {
    tot *= p_factors[i] - 1;
  }

  return tot;
}

int power(int a, int e) {
  if (e == 0) {
    return 1;
  }

  int output = 1;

  for (int i = 0; i < e; i++) {
    output *= a;
  }

  return output;
}

int main() {
  int a;
  int p;
  scanf("%d %d", &a, &p);
  if (gcd(a, p) != 1) {
    fprintf(stderr, "a: %d, b: %d are not coprimes.", a, p);
  }
  int tmp = power(a, totient(p));

  if (tmp % p != 1) {
    fprintf(stderr, "Not possible for a: %d, b: %d\n", a, p);
  } else {
    puts("Verified!");
  }
  return 0;
}
