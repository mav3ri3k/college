#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Function for modular exponentiation (a^b mod p)
long long power(long long base, long long exp, long long mod) {
    long long res = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp = exp / 2;
    }
    return res;
}

// Function to find a prime number (for simplicity, not cryptographically strong)
long long findPrime() {
    long long primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}; //small list of primes
    return primes[rand() % (sizeof(primes) / sizeof(primes[0]))];
}

// Function to find a primitive root modulo p (not fully robust, for simplicity)
long long findPrimitiveRoot(long long p) {
    // For simplicity, using 2. In real implementation, more robust algorithms are needed.
    return 2;
}

int main() {
    srand(time(0));

    // Publicly known parameters
    long long p = findPrime(); // Large prime number
    long long g = findPrimitiveRoot(p); // Primitive root modulo p

    printf("Publicly known parameters:\n");
    printf("Prime (p): %lld\n", p);
    printf("Primitive root (g): %lld\n\n", g);

    // Alice's private key (a)
    long long a = rand() % (p - 2) + 2; // Random number between 2 and p-1

    // Bob's private key (b)
    long long b = rand() % (p - 2) + 2; // Random number between 2 and p-1

    // Alice computes A = g^a mod p
    long long A = power(g, a, p);

    // Bob computes B = g^b mod p
    long long B = power(g, b, p);

    printf("Alice's private key (a): %lld\n", a);
    printf("Bob's private key (b): %lld\n\n", b);

    printf("Alice computes A: %lld\n", A);
    printf("Bob computes B: %lld\n\n", B);

    // Alice computes shared key K_A = B^a mod p
    long long KA = power(B, a, p);

    // Bob computes shared key K_B = A^b mod p
    long long KB = power(A, b, p);

    printf("Alice's shared key (KA): %lld\n", KA);
    printf("Bob's shared key (KB): %lld\n\n", KB);

    if (KA == KB) {
        printf("Shared keys match! Diffie-Hellman key exchange successful.\n");
    } else {
        printf("Shared keys do not match! Diffie-Hellman key exchange failed.\n");
    }

    return 0;
}
