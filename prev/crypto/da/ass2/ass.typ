#import "lib.typ": *


#show: doc => assignment(
  title: "Digital Assignment - II",
  course: "Cryptography and Network Security Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "16 February, 2025",
  doc,
)

= Fermat's Theorem

== Code
#code-block(
  ctitle: "main.c",
  ```c
// fermats_theorem.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Function to check if a number is prime.
bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0)
            return false;
    }
    return true;
}

// Fast modular exponentiation: computes (base^exp) mod mod.
long long modExp(long long base, long long exp, int mod) {
    long long result = 1;
    base = base % mod;
    while(exp > 0) {
        if(exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;  // divide exp by 2
        base = (base * base) % mod;
    }
    return result;
}

int main(void) {
    int a, p;
    printf("Fermat's Little Theorem Checker\n");
    printf("Enter an integer a: ");
    if (scanf("%d", &a) != 1) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }
    printf("Enter a prime number p: ");
    if (scanf("%d", &p) != 1) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }
    
    if (!isPrime(p)) {
        printf("Error: %d is not a prime number.\n", p);
        return 1;
    }
    
    if (a % p == 0) {
        printf("Note: a is divisible by p. Fermat's theorem applies only if a is not divisible by p.\n");
        return 1;
    }
    
    // Fermat's Little Theorem: a^(p-1) mod p should equal 1.
    long long result = modExp(a, p - 1, p);
    printf("Computed: %d^(%d-1) mod %d = %lld\n", a, p, p, result);
    
    if (result == 1)
        printf("Fermat's Little Theorem holds for a = %d and p = %d.\n", a, p);
    else
        printf("Fermat's Little Theorem does not hold (unexpected result).\n");
    
    return 0;
}

  ```,
)
== Output
#image("q1.png")

= Euler' Theorem

== Code
#code-block(
  ctitle: "main.c",
  ```c
// euler_theorem.c
#include <stdio.h>
#include <stdlib.h>

// Function to compute the Greatest Common Divisor using the Euclidean algorithm.
int gcd(int a, int b) {
    while(b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Fast modular exponentiation: computes (base^exp) mod mod.
long long modExp(long long base, long long exp, int mod) {
    long long result = 1;
    base = base % mod;
    while(exp > 0) {
        if(exp % 2 == 1)
            result = (result * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return result;
}

// Function to compute Euler's Totient Function, φ(n)
int phi(int n) {
    int result = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while(n % i == 0)
                n /= i;
            result -= result / i;
        }
    }
    if(n > 1)
        result -= result / n;
    return result;
}

int main(void) {
    int a, n;
    printf("Euler's Theorem Checker\n");
    printf("Enter an integer a: ");
    if(scanf("%d", &a) != 1) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }
    printf("Enter a positive integer n: ");
    if(scanf("%d", &n) != 1 || n <= 0) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }
    
    int g = gcd(a, n);
    printf("gcd(%d, %d) = %d\n", a, n, g);
    
    if(g != 1) {
        printf("Case ii: Since gcd(a, n) ≠ 1, Euler's Theorem does not apply.\n");
    } else {
        // Case i: When a and n are relatively prime.
        int totient = phi(n);
        printf("Euler's Totient Function φ(%d) = %d\n", n, totient);
        long long result = modExp(a, totient, n);
        printf("Computed: %d^(φ(%d)) mod %d = %lld\n", a, n, n, result);
        if(result == 1)
            printf("Euler's Theorem holds for a = %d and n = %d.\n", a, n);
        else
            printf("Euler's Theorem does not hold (unexpected result).\n");
    }
    
    return 0;
}
  ```,
)
== Output
#image("q2.png")

= Euclidian Algorithm

== Code
#code-block(
  ctitle: "main.c",
  ```c
// euclidean_algorithm.c
#include <stdio.h>
#include <stdlib.h>

// Euclidean Algorithm to compute gcd of two numbers.
int gcd(int a, int b) {
    while(b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int main(void) {
    int num1, num2;
    printf("Euclidean Algorithm for GCD\n");
    printf("Enter first integer: ");
    if(scanf("%d", &num1) != 1) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }
    printf("Enter second integer: ");
    if(scanf("%d", &num2) != 1) {
        fprintf(stderr, "Invalid input.\n");
        return 1;
    }
    
    int result = gcd(num1, num2);
    printf("gcd(%d, %d) = %d\n", num1, num2, result);
    
    return 0;
}

  ```,
)
== Output
#image("q3.png")
