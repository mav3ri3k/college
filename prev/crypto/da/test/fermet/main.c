#include <stdio.h>
#include <stdlib.h>

int gcd(int a, int b) {
  while(b != 0) {
    int tmp = b;
    b = a % b;
    a = tmp;
  }

  return a;
}

int is_prime(int n) {
  for (int i = 2; i < n; i++) {
    if (gcd(i, n) != 1) {
      return 0;
    }
  }

  return 1;
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

int verify(int a, int p) {
  int tmp = power(a, p-1);

  if (tmp % p == 1) {
    return 1;
  } else {
    return 0;
  }
}

int main() {
  // p is prime
  // a is not div by p
  int a;
  int p;
  puts("Input number: x and prime: p");
  scanf("%d %d", &a, &p);
;
  if (gcd(a, p) != 1) {
    fprintf(stderr, "%d and %d are not co-primes.\n", a, p);
  }

  if(!is_prime(p)) {
    fprintf(stderr, "%d is not prime\n", p);
  }

  if (verify(a, p)) {
    puts("Verified!\n");
  } else {
    printf("Not possible for given values of a: %d, p: %d\n", a, p);
  }
  
  return 0;
}
