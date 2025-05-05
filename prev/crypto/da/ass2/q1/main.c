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



