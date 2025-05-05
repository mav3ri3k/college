#include <stdio.h>

struct pub_key {
  int E;
  int N;
};

struct priv_key {
  int D;
  int N;
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

int encrypt(int msg, struct pub_key k) {
  int tmp = power(msg, k.E);

  return tmp % k.N;
}

int decrypt(int msg, struct priv_key k) {
  int tmp = power(msg, k.D);

  return tmp % k.N;
}

int main() {
  struct pub_key pub_k = {7, 33};
  struct priv_key priv_k = {3, 33};
  int msg = 2;
  printf("msg: %d\n", msg);

  int e_msg = encrypt(msg, pub_k);
  printf("e_msg: %d\n", e_msg);

  int d_msg = decrypt(e_msg, priv_k);
  printf("d_msg: %d\n", d_msg);

  
  return 0;
}
