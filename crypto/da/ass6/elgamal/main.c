#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct pub_key {
  int q;
  int alpha;
  int Y;
};

struct priv_key {
  int x;
  int q;
};

struct enc_msg {
  int c1;
  int c2;
};

void print_enc_msg(struct enc_msg E) {
  printf("Encrypted Message:\n");
  printf("  C1: %d, C2: %d\n", E.c1, E.c2);
}

int gcd(int a, int b) {
  if (a < b)
    return gcd(b, a);
  else if (a % b == 0)
    return b;
  else
    return gcd(b, a % b);
}

int gen_key(int q) {
  int key = rand() % q;
  while (gcd(q, key) != 1) {
    key = rand() % q + 91;
  }
  return key;
}

int power(int a, int b, int c) {
  int x = 1;
  int y = a;
  while (b > 0) {
    if (b % 2 != 0) {
      x = (x * y) % c;
    }
    y = (y * y) % c;
    b = b / 2;
  }
  return x % c;
}

struct enc_msg encrypt(int msg, struct pub_key P) {
  if (msg > P.q) {
    perror("Msg cant not be more than q");
    exit(0);
  }

  int k = rand() % P.q;

  int K = power(P.Y, k, P.q);

  int c1 = power(P.alpha, k, P.q);
  int c2 = (K * msg) % P.q;

  struct enc_msg E = {c1, c2};

  return E;
}

int modInverse(int a, int p) {
  if (p < 0) {
    perror("The prime cna not be negative");
    exit(0);
  }

  int flag = 0;
  if (a < 0) {
    a = -a;
    flag = 1;
  }

  int i = 0;
  while (i < 100) {
    if ((a*i) % p == 1) {
      break;
    }

    i += 1; 
  }

  if (flag) {
    return p - i;
  } else {
    return i;
  }
}

int decrypt(struct enc_msg E, struct priv_key P) {
  int K = power(E.c1, P.x, P.q);
  int tmp = modInverse(K, P.q);

  int msg = (E.c2 * tmp) % P.q;

  return msg;
}

int main() {
  int msg;
  printf("Enter the message: ");
  scanf("%d", &msg);
  int q = rand() % 91;
  int alpha = rand() % (q - 2) + 2;
  int X = gen_key(q);

  struct priv_key priv_key = {X, q};

  int Y = power(alpha, X, q);
  struct pub_key pub_key= {q, alpha, Y};

  struct enc_msg E = encrypt(msg, pub_key);

  print_enc_msg(E);

  int dec_msg = decrypt(E, priv_key);

  printf("Decrypted Message: %d\n", dec_msg);
  return 0;
}
