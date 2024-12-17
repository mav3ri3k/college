%option noyywrap
%{
#include <stdio.h>
%}
url (www)\.((([a-zA-Z0-9])|\_|\.)*)\.((in)|(com)|(org))
invalid .*
%%
{url} {printf("Valid URL");}
{invalid} {printf("Invalid URL");}
%%
int main() {
 yyin = fopen("input.txt", "r");
 yylex();
 return 0;
}
