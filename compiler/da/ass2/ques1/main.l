%option noyywrap
%{
#include <stdio.h>
%}
email ([a-z0-9A-Z])(([a-z0-9A-Z]|\.)*)@([a-zA-Z0-9]+)\.([a-zA-Z0-9]+)
invalid .*
%%
{email} {printf("Valid Email");}
{invalid} {printf("Invalid Email");}
%%
int main() {
 yyin = fopen("input.txt", "r");
 yylex();
 return 0;
}
