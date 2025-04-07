#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

// Function to check if a number is prime (using a basic method)
bool is_prime(long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

// Function to generate a random prime number within a specified range.
long long chose_prim() {
    long long min = 1000; // Example minimum range
    long long max = 2000; // Example maximum range

    if (min < 2) {
        min = 2;
    }

    long long randomNum;
    srand(time(NULL));

    while (1) {
        randomNum = (rand() % (max - min + 1)) + min;
        if (is_prime(randomNum)) {
            return randomNum;
        }
    }
}

// Function to calculate the greatest common divisor (GCD)
long long gcd(long long a, long long b) {
    while (b != 0) {
        long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Function to choose a valid 'e' (public exponent)
long long chose_E(long long phi) {
    long long e = rand() % (phi - 2) + 2; // Ensure e is between 2 and phi-1

    while (gcd(e, phi) != 1) {
        e = rand() % (phi - 2) + 2;
    }
    return e;
}

// Function to calculate the modular inverse
long long mod_inverse(long long e, long long phi) {
    long long d = 1;
    while ((e * d) % phi != 1) {
        d++;
    }
    return d;
}

// Function to encrypt the message
long long encrypt(long long msg, long long e, long long n) {
    long long result = 1;
    msg = msg % n;
    while (e > 0) {
        if (e % 2 == 1) {
            result = (result * msg) % n;
        }
        msg = (msg * msg) % n;
        e = e / 2;
    }
    return result;
}

// Function to decrypt the message
long long decrypt(long long ciphertext, long long d, long long n) {
    long long result = 1;
    ciphertext = ciphertext % n;
    while (d > 0) {
        if (d % 2 == 1) {
            result = (result * ciphertext) % n;
        }
        ciphertext = (ciphertext * ciphertext) % n;
        d = d / 2;
    }
    return result;
}

int main(void) {
    srand(time(0));

    long long p = chose_prim();
    long long q = chose_prim();

    while (q == p) {
        q = chose_prim();
    }

    long long n = p * q;
    long long phi = (p - 1) * (q - 1);

    long long e = chose_E(phi);
    long long d = mod_inverse(e, phi);

    long long msg = 0;
    printf("Enter msg to encrypt: ");
    scanf("%lld", &msg);

    printf("p: %lld, q: %lld, n: %lld, phi: %lld, e: %lld, d: %lld\n", p, q, n, phi, e, d);

    long long ciphertext = encrypt(msg, e, n);
    printf("Cypher Text: %lld\n", ciphertext);

    long long decrypted_msg = decrypt(ciphertext, d, n);
    printf("Decrypted Text: %lld\n", decrypted_msg);

    return 0;
}
