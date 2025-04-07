#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 256

struct Point {
  int x;
  int y;
};

void print_point(struct Point p) {
  printf("Point:\n");
  printf("  X: %d\n  Y: %d\n", p.x, p.y);
}

struct Curve {
  int a;
  int b;
  int p;
};

void print_curve(struct Curve c) {
  printf("Curve:\n");
  printf("  a: %d\n  b: %d\n  p: %d\n", c.a, c.b, c.p);
}

int gcd(int a, int b)
{
    // Find Minimum of a and b
    int result = ((a < b) ? a : b);
    while (result > 0) {
        // Check if both a and b are divisible by result
        if (a % result == 0 && b % result == 0) {
            break;
        }
        result--;
    }
    // return gcd of a nd b
    return result;
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

int mod(int num, int p) {
  if (p < 0) {
    perror("The prime can not be negative");
    exit(0);
  }
  if (num < 0) {
    int tmp = (-num) % p;
    return p - tmp;
  } else {
    return num % p;
  }
 }

 int mod_frac(int num, int den, int p) {
   int a = mod(num, p);
   int b = modInverse(den, p);

   return (a*b) % p;
 }

struct Point ecc_double(struct Point P, struct Point Q, struct Curve E) {
  int lambda = 0;
  if (P.x == Q.x && P.y == Q.y) {
   lambda = mod_frac((3*P.x*P.x)  + E.a, (2*P.y), E.p);
  } else {
    lambda = mod_frac((Q.y - P.y), (Q.x - P.x), E.p);
  }


  struct Point R = {0};

  int tmp_x = (lambda*lambda) - P.x - Q.x;
  R.x = mod(tmp_x, E.p);

  int tmp_y = (lambda*(P.x - R.x)) - P.y;
  R.y = mod(tmp_y, E.p);

  return R;
}

int main(void) {
  struct Curve E = {0};
  struct Point P = {0};
  struct Point Q = {0};

  printf("Inputs: \n");
  printf("Curve: a b p: ");
  scanf("%d %d %d", &E.a, &E.b, &E.p);

  printf("Point P: x y: ");
  scanf("%d %d", &P.x, &P.y);
  printf("Point Q: x y: ");
  scanf("%d %d", &Q.x, &Q.y);

  print_curve(E);
  print_point(P);
  print_point(Q);

  struct Point R = ecc_double(P, Q, E);

  printf("P + Q = \n");
  print_point(R);
  return 0;
}
