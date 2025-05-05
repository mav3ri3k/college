#include <stdio.h>
#include <stdlib.h>

struct Point {
  int x;
  int y;
  int a;
  int m;
};

void print_point(struct Point p) {
  puts("Points: ");
  printf("  x: %d, y: %d, m: %d\n", p.x, p.y, p.m);
}

int power(int a, int e) {
  if (e == 0) {
    return 1;
  }

  int output = 1;
  for (int i = 0; i < e; i++) {
    output *=a ;
  };

  return output;
}

int mod(int in, int m) {
  int flag = 0;
  if (in < 0) {
    in = -in;
    flag = 1;
  }

  int out = in % m;

  if (flag) {
    return m-out;
  } else {
    return out;
  }
}

int mod_inverse(int x, int m) {
  int ans = 1;
  int i = 0;
  while ((x*ans)%m != 1 && ans < 500) {
    ans += 1;
  }

  if (i > 500) {
    fprintf(stdin, "Could not find mod inverse for x: %d, m: %d", x, m);
    exit(0);
  }

  return ans;
}

void double_ecc(struct Point *p) {
  int num = mod(((3*p->x*p->x) + p->a),p->m); 
  int den = mod_inverse(2*p->y, p->m);


  int lambda = mod((num*den), p->m);

  int x = (lambda*lambda) - (2*p->x);
  int y = (lambda*(p->x - x)) - p->y;


  p->x = mod(x, p->m);
  p->y = mod(y, p-> m);
}


int main() {
  struct Point p = {3, 10, 1, 23};
  print_point(p);

  double_ecc(&p);
  print_point(p);
  return 0;
}
