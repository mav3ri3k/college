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
  title: "Digital Assignment - II",
  course: "Compiler Design Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  authors: ("Apurva Mishra, 22BCE2791",),
  date: "19 August 2024",
  doc,
)

#problem[
  Write a LEX code to check email id is valid or not.

  #code_block(
    ctitle: "Code",
    ```c
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
    ```,
  )

  #code_block(
    ctitle: "Input Text",
    ```markdown
    apurva.jpr@gmail.com
    apurva.jpr@yahoo.com
    valid@email.com

    invalid@emailcom
    .invliad@gmail.com
    ```,
  )

]
#pagebreak()
#text(weight: "bold", size: 15pt)[Output]
#image("q1.png")

#problem[
  Write a LEX code to check URL is valid or not

  #code_block(
    ctitle: "Code",
    ```c
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
    ```,
  )

  #code_block(
    ctitle: "Input Text",
    ```markdown
    www.apurva.com
    www.google.com
    www.evilcorp.org
    www.mycountry.in
    www.goo_.com
    www.vit.ac.in

    www.$money.com
    sheets.google.com
    vtop.vit.ac
    google.com
    www.google.somelongtext
    www.google
    www.google?.com
    ```,
  )

]

#pagebreak()
#text(weight: "bold", size: 15pt)[Output]
#image("q2.png")


#problem[
  Write a LEX code to print the length of longest word in a file.

  #code_block(
    ctitle: "Code",
    ```c
    %option noyywrap
    %{
    #include <stdio.h>
    int len = 0;
    %}
    word ([a-zA-Z])+
    invalid .*
    %%
    {word} {printf("Valid World");len=len>yyleng?len:yyleng;}
    {invalid} {printf("Invalid Word");}
    %%
    int main() {
     yyin = fopen("input.txt", "r");
     yylex();
     printf("Length of longest word: %d", len);
     return 0;
    }
    ```,
  )

  #code_block(
    ctitle: "Input Text",
    ```markdown
    These
    are
    some
    verrry
    long
    words
    Actually
    No
    ```,
  )

]

#pagebreak()
#text(weight: "bold", size: 15pt)[Output]
#image("q3.png")

