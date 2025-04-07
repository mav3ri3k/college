#include <stdio.h>

struct values {
  int q;
  int g;
};

struct person {
  int priv;
  int pub;
  int key;
};

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
  struct values v = {11, 7};
  struct person A = {5, 0, 0};
  struct person B = {3, 0, 0};

  A.pub = power(v.g, A.priv) % v.q;
  B.pub = power(v.g, B.priv) % v.q;

  A.key = power(B.pub, A.priv) % v.q;
  B.key = power(A.pub, B.priv) % v.q;

  printf("A: %d, B: %d\n", A.key, B.key);
  return 0;
}
