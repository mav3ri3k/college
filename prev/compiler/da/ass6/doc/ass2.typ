#import "lib.typ": *
#import "dvd.typ": *
#import "@preview/showybox:2.0.1": showybox

#show image: it => block(
  radius: 10pt,
  clip: true,
)[#it]

#let code_block(ctitle: "Here", cbody) = {
  showybox(
    frame: (
      border-color: black,
      title-color: red.lighten(60%),
      body-color: luma(230),
    ),
    title-style: (
      color: black,
      weight: 100pt,
    ),
    body-style: (
      align: left,
    ),
    sep: (
      dash: "dashed",
    ),
    shadow: (
      offset: (x: 1pt, y: 1pt),
      color: black.lighten(70%),
    ),
    breakable: true,
    text(weight: "bold")[#ctitle],
    cbody,
  )
}
#show raw: name => if name.block [
  #name
] else [
  #box(
    fill: luma(230),
    outset: (x: 2pt, y: 3pt),
    radius: 4pt,
  )[#name]
]

#show: doc => report(
  title: "Digital Assignment - VI",
  course: "Compiler Design Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  authors: ("Apurva Mishra, 22BCE2791",),
  date: "12 Novemeber 2024",
  doc,
)

#text(weight: "bold")[Note: This is mirror of Assessment 6 submitted on 12 November, 2024 on Moodle]
= Question

#problem[
  To write a YACC program to recognize strings of { b#super[n]aa | n>=5}.
]
#code_block(
  ctitle: "main.l",
  ```c
  %{
  #include <stdlib.h>
  #include <stdio.h>
  void yyerror(char *);
  #include "y.tab.h"
  %}
  %%
  ((bbbbb)(b*)(aa)) { return PATTERN; }
  [ \t] ;
  . yyerror("invalid character");
  %%
  int yywrap(void) {
  return 1;
  }
  ```,
)

#code_block(
  ctitle: "main.y",
  ```c
  %token PATTERN
  %{
  #include <stdio.h>
  void yyerror(char *);
  int yylex(void);
  %}
  %%
  program:
  program expr '\n'
  |
  ;
  expr:
  PATTERN {$$ = 1; printf("Accepted");}
  |
  ;
  %%
  void yyerror(char *s) {
  fprintf(stderr, "%s\n", s);
  }
  int main(void) {
  yyparse();
  return 0;
  }
  ```,
)

#code_block(
  ctitle: "run.sh",
  ```bash
  #!/bin/bash

  lex main.l
  yacc -d main.y
  gcc lex.yy.c y.tab.c -o main
  ./main
  ```,
)

#text(weight: "bold", size: 15pt)[Output]
#image("q1.png")

= Question

#problem[
  Write a C program to implement Code optimization technique.
]
#code_block(
  ctitle: "main.l",
  ```c
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
  ```,
)

#code_block(
  ctitle: "main.y",
  ```c
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
  ```,
)

#code_block(
  ctitle: "core.h",
  ```c
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
  ```,
)

#code_block(
  ctitle: "run.sh",
  ```bash
  #!/bin/bash

  lex main.l
  yacc -d main.y
  gcc lex.yy.c y.tab.c -o main
  ./main
  ```,
)

#text(weight: "bold", size: 15pt)[Output]
\
The given code applies following two optimisations:
- Common Subexpressions Elimination
- Constant Folding

#image("q2.png")

