%option noyywrap
%{
#include <stdio.h>
int count = 0;
int def = 0;
int con = 0;
%}
feb ((((0|1))([0-9]))|((2)([0-8])))("/")(((0([0-9]))|(1([0-2]))))("/")()
one ((((0|1|2))([0-9]))|((3)([0-1])))("/")(((0([0-9]))|(1([0-2]))))("/")
two ((((0|1|2))([0-9]))|((3)([0])))("/")(((0([0-9]))|(1([0-2]))))("/")
invalid (.)*
%%
{feb} {printf("Valid");}
{one} {printf("Valid");}
{two} {printf("Valid");}
{invalid} {printf("Invalid: %s", yytext);}
%%
int main() {
 yyin = fopen("input.txt", "r");
 yylex();
 printf("Count of define statement: %d\n", def);
 printf("Count of cont statement: %d\n", con);
 printf("Count of constants: %d\n", def+con);
 return 0;
}
