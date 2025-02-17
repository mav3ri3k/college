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
