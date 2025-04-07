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

int mod(int num, int p) {
  if (p < 0) {
    perror("The prime can not be negative");
    exit(0);
  }
  if (num < 0) {
    printf("%d\n", num);
    int tmp = (-num) % p;
    printf("%d %d\n", p, tmp);
    return p - tmp;
  } else {
    return num % p;
  }
 }

struct Point ecc_neg(struct Point P, struct Curve E) {
  struct Point R = {0};

  R.x = P.x;
  R.y = mod(-P.y, E.p);

  return R;
}

int main(void) {
  struct Curve E = {0};
  struct Point P = {0};

  printf("Inputs: \n");
  printf("Curve: a b p: ");
  scanf("%d %d %d", &E.a, &E.b, &E.p);

  printf("Point P: x y: ");
  scanf("%d %d", &P.x, &P.y);

  print_curve(E);
  print_point(P);

  struct Point R = ecc_neg(P, E);

  printf("\n-P = \n");
  print_point(R);
  return 0;
}
