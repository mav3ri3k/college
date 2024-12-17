%option noyywrap
%{
#include <stdio.h>
int a = 0;
int b = 0;
int op = 0;
void digi() {
if (op==0) {
a = atof(yytext);
} else {
b=atof(yytext);
switch(op){
case 1:
a += b;
break;
case 2:
a -= b;
break;
}
printf("Ans: %d", a);
}
op = 0;
}
%}
dig ([0-9]+)
add "+"
sub "-"
%%
{dig} {digi();}
{add} {op=1;}
{sub} {op=2;}
%%
int main() {
 yyin = fopen("input.txt", "r");
 yylex();
 return 0;
}
