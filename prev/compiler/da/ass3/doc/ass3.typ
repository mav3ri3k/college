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
  title: "Digital Assignment - III",
  subtitle: "Compiler Design Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  authors: ("Apurva Mishra, 22BCE2791",),
  date: "10 September 2024",
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

#text(weight: "bold")[Note: This is mirror of Assessment 3 submitted on 9 September, 2024 on Moodle]
= Questions

#problem[
  Write a LEX code to check date is valid or not.
]

#code_block(
  ctitle: "main.l",
  ```c
  %option noyywrap
  %{
  int i = 0;
  %}

  feb (((0|1)([0-9]))|((2)([0-8])))("/")(01)("/")(([0-2])([0-9]{3}))
  one (((0|1|2)([0-9]))|((3)([0-1])))("/")("01"|"03"|"05"|"07"|"08"|"10"|"12")("/")(([0-2])([0-9]{3})) twoo (((0|1|2)([0-9]))|((3)(0)))("/")("04"|"06"|"09"|"11")("/")(([0-2])([0-9]{3}))
  invalid (.)*

  %%
  {feb} {printf("Valid: %s", yytext);}
  {one} {printf("Valid: %s", yytext);}
  {twoo} {printf("Valid: %s", yytext);}
  {invalid} {printf("Invalid: %s", yytext);}
  %%

  int main() {
  yyin = fopen("input.txt", "r"); yylex();
  return 0;
  }
  ```,
)

#text(size: 15pt, weight: "bold")[Output]
#image("q1.png")

#pagebreak()
#problem[
  Write a LEX code to count total number of tokens in a given C File.
]

#code_block(
  ctitle: "main.l",
  ```c
  %option noyywrap

  %{
  int ct = 0;
  %}

  keyword ([a-zA-Z])(([a-zA-Z0-9])*) cont ([0-9])+
  op ("=="|">"|"<"|"=")
  del (","|";"|"("|")")
  invalid (.)*

  %%
  {keyword} {ct += 1;}
  {cont} {ct += 1;}
  {op} {ct += 1;}
  {del} {ct += 1;}
  {invalid} {printf("Invalid Token: %s", yytext);}
  %%

  int main() {
  yyin = fopen("input.txt", "r"); yylex();
  printf("Count: %d", ct); return 0;
  }
  ```,
)

#text(size: 15pt, weight: "bold")[Output]
#image("./q2.png")

#problem[
  Write a LEX code to count the frequency of the given word in a file
]


#code_block(
  ctitle: "main.l",
  ```c
  %option noyywrap %{
  int ct = 0;
  void count() {
  int len = yyleng;
  char token[] = "some"; int tp = 0;
  for (int i = 0; i < len; i++) {
  while (token[tp] == yytext[i+tp] && tp < 4) { tp += 1;
  }
  if (tp == 4) {
  ct += 1;
  }
  tp = 0;
  }
  }
  %}

  invalid (.)*

  %%
  {invalid} {count();}
  %%

  int main() {
  yyin = fopen("input.txt", "r");
  yylex();
  printf("Frequency: %d", ct);
  return 0;
  }
  ```,
)
#text(size: 15pt, weight: "bold")[Output]
#image("./q3.png")
