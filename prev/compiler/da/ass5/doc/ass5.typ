#import "lib.typ": *
#import "@preview/dvdtyp:1.0.0": *

#show image: it => block(
  radius: 10pt,
  clip: true,
)[#it]

#show raw: name => if name.block [
  #block(
    fill: luma(230),
    inset: 4pt,
    radius: 10pt,
  )[#name]
] else [
  #box(
    fill: luma(230),
    outset: (x: 2pt, y: 3pt),
    radius: 10pt,
  )[#name]
]

#show: doc => report(
  title: "Digital Assignment - V",
  subtitle: "Compiler Design Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  authors: ("Apurva Mishra, 22BCE2791",),
  date: "21 October 2024",
  doc,
)

#let view(
  question,
  output,
  output_size,
  raw_code,
) = {
  problem[
    #question
  ]
  grid(
    inset: 3pt,
    columns: (auto, auto),
    /*
    align(center)[
      #image(output, height: output_size, fit: "stretch")
    ],
    */
    raw_code,
  )
}

#text(weight: "bold")[Note: This is mirror of Assessment 5 submitted on 21 October, 2024 on Moodle]
= Questions

#problem[
  Write a C program to generate 3-Address code for a given expression
]

#text(size: 15pt, weight: "bold")[Code]
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
  [-+/*\n] { return *yytext; }
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
  token INTEGER VARIABLE
  %left '+' '-'
  %left '*' '/'
  %{
  #include <stdio.h>
  void yyerror(char *);
  int yylex(void);
  int sym[26];
  int count = 0;
  %}
  %%
  program:
  program statement '\n'
  |
  ;
  statement:
  expr { ;}
  |
  ;
  expr:
  INTEGER
  | expr '+' expr { $$ = $1 + $3; if (count == 0) {printf("t%d = %d + %d\n", count, $1, $3);} else {printf("t%d = t%d + %d\n", count, count-1, $3);} count++;}
  | expr '-' expr { $$ = $1 - $3; if (count == 0) {printf("t%d = %d - %d\n", count, $1, $3);} else {printf("t%d = t%d - %d\n", count, count-1, $3);} count++;}
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
#text(size: 15pt, weight: "bold")[Output]
#image("q1.png")

#pagebreak()
#problem[
  Write a C Program to implement Type Checking
]

#text(size: 15pt, weight: "bold")[Code]
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
  ("int") {return TINTEGER; }
  ("char") {return TCHAR; }
  [0-9]+ {
  yylval = atoi(yytext);
  return INTEGER;
  }
  [a-z]+ { return WORD; }
  [=] { return *yytext; }
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
  %token TINTEGER TCHAR INTEGER WORD
  %left '='
  %{
  #include <stdio.h>
  void yyerror(char *);
  int yylex(void);
  int sym[26];
  %}
  %%
  program:
  program statement '\n'
  |
  ;
  statement:
  expr { ;}
  |
  ;
  expr:
  INTEGER
  | TINTEGER WORD'=' INTEGER {$$ = 1;printf("Valid type");}
  | TCHAR WORD '=' WORD {$$ = 1;printf("Valid type");}
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
  ```c
  #!/bin/bash
  lex main.l
  yacc -d main.y
  gcc lex.yy.c y.tab.c -o main
  ./main
  ```,
)
#text(size: 15pt, weight: "bold")[Output]
#image("./q2a.png")
#image("./q2b.png")
#image("./q2c.png")
