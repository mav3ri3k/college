%option noyywrap
%{
#include <stdio.h>
int len = 0;
%}
word ([a-zA-Z])+
invalid .*
%%
{word} {printf("Valid Word");len=len>yyleng?len:yyleng;}
{invalid} {printf("Invalid Word");}
%%
int main() {
 yyin = fopen("input.txt", "r");
 yylex();
 printf("Length of longest word: %d", len);
 return 0;
}
