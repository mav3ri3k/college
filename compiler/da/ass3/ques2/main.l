%option noyywrap
%{
#include <stdio.h>
#include <string.h>
int count = 0;
int def = 0;
%}
word ([a-zA-Z]+)
invalid (.)*
%%
{word} {strcmp(yytext, "apurva")==0?count++:def++;}
{invalid} {printf("Invalid: %s", yytext);}
%%
int main() {
 yyin = fopen("input.txt", "r");
 yylex();
 printf("Count : %d\n", count);
 return 0;
}
