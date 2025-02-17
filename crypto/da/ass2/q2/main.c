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
