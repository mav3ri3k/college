#import "lib.typ": report
#import "@preview/dvdtyp:1.0.0": *


#show raw: name => if name.block [
  #block(
    fill: luma(230),
    inset: 4pt,
    radius: 4pt,
  )[#name]
] else [
  #box(
    fill: luma(230),
    outset: (x: 2pt, y: 3pt),
    radius: 4pt,
  )[#name]
]

#show: doc => report(
  title: "Digital Assignment - I",
  subtitle: "Operating System Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  authors: ("Apurva Mishra, 22BCE2791",),
  date: "2 August 2024",
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
    align(center)[
      #image(output, height: output_size, fit: "stretch")
    ],
    raw_code,
  )
}

= Questions

#view(
  "Create a shell script program to determine whether or not an input number is a palindrome.",
  "q1.png",
  85%,

  ```bash
  #!/bin/bash

  is_palindrome() {
    num=$1
    reversed=$(echo $num | rev)

    if [ "$num" -eq "$reversed" ]; then
      echo "The number $num is a palindrome"
    else
      echo "The number $num is not a palindrome"
    fi
  }

  read -p "Enter a number: " num

  is_palindrome $num
  ```,
)


#problem[
  Create a shell script program to determine whether or not an input number is a palindrome.
]
#align(center)[
  #image("./q1.png", height: 87%)
]

#problem[
  Create a shell script program to add n numbers.
]

#grid(
  inset: 4pt,
  columns: (auto, auto),
  align(center)[
    #image("./q2.png", height: 89%)
  ],
  ```bash
  #!/bin/bash

  sum=0
  echo "Number of numbers to add:"
  read n

  echo "Enter the numbers:"

  for ((i=1; i<=n; i++))
  do
    read num
    sum=$((sum + num))
  done

  echo "The sum is: $sum"
  ```,
)

#problem[
  Create a shell script program to add array elements.
]
#align(center)[
  #image("./q3.png", height: 85%)
]
