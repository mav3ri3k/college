%{
#include <stdlib.h>
#include <stdio.h>
void yyerror(char *);
#include "y.tab.h"
%}
%%
[0-9]+ {
yylval = atoi(yytext);
return INTEGER;
}
[a-z] { 
yylval = *yytext;
return VARIABLE; 
}
("jmp") { return JMP; }
[-+/*;=] { return *yytext; }
[ \t\n]+ ;
. yyerror("invalid character");
%%
int yywrap(void) {
return 1;
}
