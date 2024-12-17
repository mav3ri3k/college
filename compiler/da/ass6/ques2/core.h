#include <stdio.h>

struct eval {
    int var;
    int var1;
    int var2;
};

struct eval global[10];
int i = 0;

int skip_count = 0;

void new_node(int var, int var1, int var2) {
    global[i].var = var;
    global[i].var1 = var1;
    global[i].var2 = var2;

    i += 1;
}

void print_line() {
    if (skip_count > 0) {
        skip_count -= 1;
        return;
    }

    int k = i - 1;

    int prev = -1;
    for (int j = 0; j < k; j++) {
        if (global[j].var1 == global[k].var1 && global[j].var2 == global[k].var2) {
            prev = global[j].var;
            break;
        }
    }

    if (prev != -1) {
        printf("%c = %c; | Common Subexpressions Elimination\n", global[k].var, prev);
        return;
    }

    if (global[k].var1 >= 'a' && global[k].var2 >= 'a') {
        printf("%c = %c + %c; | No optimization\n", global[k].var, global[k].var1, global[k].var2);
    } else if (global[k].var1 >= 'a' && global[k].var2 < 'a') {
        printf("%c = %c + %d; | No optimization\n", global[k].var, global[k].var1, global[k].var2);
    } else if (global[k].var1 < 'a' && global[k].var2 >= 'a') {
        printf("%c = %d + %c; | No optimization\n", global[k].var, global[k].var1, global[k].var2);
    } else {
        printf("%c = %d; | Constant Folding\n", global[k].var, global[k].var1 + global[k].var2);
    }
}
