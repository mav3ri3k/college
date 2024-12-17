%token INTEGER VARIABLE JMP
%left '+' '-'
%left '*' '/'
%{
#include "core.h"
void yyerror(char *);
int yylex(void);
int sym[26];
%}
%%
program:
function { ; }
;
function:
function expr { ;}
| /* NULL */
;
expr:
| VARIABLE '=' INTEGER '+' INTEGER ';' { new_node($1, $3, $5); print_line();}
| VARIABLE '=' VARIABLE '+' INTEGER ';' { new_node($1, $3, $5); print_line();}
| VARIABLE '=' INTEGER '+' VARIABLE ';' { new_node($1, $3, $5); print_line();}
| VARIABLE '=' VARIABLE '+' VARIABLE ';' { new_node($1, $3, $5); print_line();}
| jump ';' { ; }
;
jump:
JMP INTEGER { ;}
;
%%
void yyerror(char *s) {
fprintf(stderr,"%s\n", s);
}
int main(void) {
yyparse();
return 0;
}
