#include <stdio.h>

struct Point {
  int x;
  int y;
  int m;
};

void print_point(struct Point *p) {
  puts("Points:");
  printf("  x: %d, y: %d, m: %d\n", p->x, p->y, p->m);
}

int neg_modulo(int input, int m) {
  int flag = 0;
  if (input < 0) {
    flag = 1;
    input = -input;
  }

  int mod = input % m;

  if (flag) {
    return m - mod;
  } else {
    return mod;
  }
}

void neg_ecc(struct Point *p) {
  p -> y = neg_modulo(-p->y,p-> m); 
}

int main() {
  struct Point p = {12, 4, 11};
  print_point(&p);

  neg_ecc(&p);
  print_point(&p);
  return 0;
}
