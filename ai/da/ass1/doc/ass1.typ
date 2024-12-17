#import "lib.typ": *
#import "@preview/dvdtyp:1.0.0": *
#import "@preview/fletcher:0.5.1": *

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
  title: "Digital Assignment - I",
  subtitle: "Artficial Intelligence",
  // Se apenas um autor colocar , no final para indicar que é um array
  authors: ("Apurva Mishra, 22BCE2791",),
  date: "10 October 2024",
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


= Questions

#problem[
  The 15 Game: To play the 15 game, write the numbers 1 through 9 in a square. Player one is X
  and player two is O. Player one starts by drawing an X through any unchosen number. Then
  player two circles an unchosen number. The game is won when one player has collected three
  numbers which add up to 15. In the sample game, player two has won because 8 + 6 + 1 = 15.
  Note that having 9 and 6 alone would not be a win even though 9 + 6 = 15, because you need
  three numbers which sum to 15.
]

#text(size: 16pt, weight: "bold")[
  \
  Q1. Draw the game tree for this game.
]

#image("q1.png", width: 200pt, height: 200pt, fit: "contain")

#pagebreak()

#text(weight: "bold")[
  Following is one of the several game trees which are
  possible for the given snapshot of the game
]
\
Dashed Lines represent possible moves which iteratively decreases as the
game continues. Line arrows represent played move

#let nodes = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
#let p11edges = (
  (0, 0),
  (0, 1),
  (0, 2),
  (0, 3),
  (0, 5),
  (0, 6),
  (0, 7),
  (0, 8),
)
#let p21edges = (
  (0, 0),
  (0, 1),
  (0, 2),
  (0, 3),
  (0, 5),
  (0, 6),
  (0, 8),
)
#let p12edges = (
  (0, 0),
  (0, 1),
  (0, 2),
  (0, 5),
  (0, 6),
  (0, 8),
)
#let p22edges = (
  (0, 0),
  (0, 1),
  (0, 2),
  (0, 6),
  (0, 8),
)
#let p13edges = (
  (0, 0),
  (0, 1),
  (0, 6),
  (0, 8),
)
#let p23edges = (
  (0, 0),
  (0, 6),
  (0, 8),
)
#let p14edges = (
  (0, 0),
  (0, 6),
)
#let p24edges = ((0, 6),)
#text(size: 12pt)[
  #diagram(

    node-stroke: .1em,
    edge-stroke: .07em,
    {

      node((4, -2), "Start", name: "Start")
      node((-1, 0), "Player 1")
      //p1 1 5
      for (i, n) in nodes.enumerate() {
        node((i, 0), n, stroke: 0pt, name: "p11" + str(i))
      }
      for (from, to) in p11edges {
        edge(label("Start"), label("p11" + str(to)), "--", bend: 0deg)
      }
      edge(label("Start"), label("p114"), "-|>", bend: 0deg)

      //p2 1 8
      node((-1, 1), "Player 2")
      for (i, n) in nodes.enumerate() {
        node((i, 1), n, stroke: 0.5pt, name: "p21" + str(i))
      }
      for (from, to) in p21edges {
        edge(label("p114"), label("p21" + str(to)), "--", bend: 0deg)
      }
      edge(label("p114"), label("p217"), "-|>", bend: 0deg)

      //p1 2 4
      node((-1, 2.5), "Player 1")
      for (i, n) in nodes.enumerate() {
        node((i, 2.5), n, stroke: 0pt, name: "p12" + str(i))
      }

      for (from, to) in p12edges {
        edge(label("p217"), label("p12" + str(to)), "--", bend: 0deg)
      }
      edge(label("p217"), label("p123"), "-|>", bend: 0deg)

      //p2 2 6
      node((-1, 4), "Player 2")
      for (i, n) in nodes.enumerate() {
        node((i, 4), n, stroke: 0.5pt, name: "p22" + str(i))
      }

      for (from, to) in p22edges {
        edge(label("p123"), label("p22" + str(to)), "--", bend: 0deg)
      }
      edge(label("p123"), label("p225"), "-|>", bend: 0deg)

      //p1 3 3
      node((-1, 5), "Player 1")
      for (i, n) in nodes.enumerate() {
        node((i, 5), n, stroke: 0pt, name: "p13" + str(i))
      }

      for (from, to) in p13edges {
        edge(label("p225"), label("p13" + str(to)), "--", bend: 0deg)
      }
      edge(label("p225"), label("p132"), "-|>", bend: 0deg)

      //p2 3 3
      node((-1, 6.5), "Player 2")
      for (i, n) in nodes.enumerate() {
        node((i, 6.5), n, stroke: 0.5pt, name: "p23" + str(i))
      }

      for (from, to) in p23edges {
        edge(label("p132"), label("p23" + str(to)), "--", bend: 0deg)
      }
      edge(label("p132"), label("p231"), "-|>", bend: 0deg)
      //
      //p1 4 9
      node((-1, 8), "Player 1")
      for (i, n) in nodes.enumerate() {
        node((i, 8), n, stroke: 0pt, name: "p14" + str(i))
      }

      for (from, to) in p23edges {
        edge(label("p231"), label("p14" + str(to)), "--", bend: 0deg)
      }
      edge(label("p231"), label("p148"), "-|>", bend: 0deg)

      //p1 4 9
      node((-1, 10), "Player 2")
      for (i, n) in nodes.enumerate() {
        node((i, 10), n, name: "p24" + str(i))
      }

      for (from, to) in p14edges {
        edge(label("p148"), label("p24" + str(to)), "--", bend: 0deg)
      }
      edge(label("p148"), label("p240"), "-|>", bend: 0deg)
    },
  )
]
\
#text(size: 16pt, weight: "bold")[
  Player 2 Won!
]
#pagebreak()

#text(size: 16pt, weight: "bold")[
  \
  Q2. What is the first best move for player one?\
]
To calculate the best move, we find all the possible combination of numbers which sum to 15

#code_block(
  ctitle: "Code to find solution combinations",

  ```rust
  //Programming Language: Rust
  use std::vec::Vec;

  fn dup(arr: Vec<u16>) -> bool {
      for i in 0..arr.len() {
          for j in 0..arr.len() {
              if i != j && arr[i] == arr[j] {
                  return true;
              }
          }
      }

      false
  }

  fn rec(i: u16, mut arr: Vec<u16>, cap: &mut Vec<u16>) {
      let sum: u16 = arr.iter().sum();
      if sum == 15 && arr.len() == 3 && !dup(arr.clone()) {
          for i in arr {
              cap[(i - 1) as usize] += 1;
              print!("{i} ");
          }
          println!();

          return;
      }

      if arr.len() == 3 {
          return;
      }

      for j in i..=9 {
          arr.push(j);
          rec(j, arr.clone(), cap);
          arr.pop();
      }
  }

  fn main() {
      let mut arr = Vec::with_capacity(3);
      let mut cap: Vec<u16> = Vec::with_capacity(9);

      for _ in 0..9 {
          cap.push(0);
      }

      println!("Possible combinations between 1-9 with sum = 15: ");
      for j in 1..=9 {
          arr.push(j);
          rec(j, arr.clone(), &mut cap);
          arr.pop();
      }

      println!();
      println!("Occurence frequency of each number: ");
      println!("[Number: Frequency]");

      for (i, v) in cap.iter().enumerate() {
          print!("[{}: {v}], ", i + 1);
      }
  }
  ```,
)

#image("q12.png")

From here we can observe that 5 has highest frequency in probable solution combinations.
Therefore 5 should be the best first move.

#problem[
  Solve the following problem

  #image("q2.png")
]
#underline(
  text(size: 17pt, weight: "bold")[Backward Chaining],
)
\
\
#text(size: 15pt, weight: "semibold")[Step 1: Conlusion]
\
#diagram(

  edge-stroke: .07em,
  {

    node((0, 0), [$7 lt.eq.slant 3 + 9$])
  },
)
\
\
#text(size: 15pt, weight: "semibold")[Step 2: Rules]
\
#diagram(

  edge-stroke: .07em,
  {

    node((0, 0), [$7 lt.eq.slant 3 + 9$])

    edge((0, 0), (0, 2), "-|>")
    node(
      (0, 2),
      [
        $forall w, x, y, z #h(0.2cm) (w
    lt.eq.slant y and x lt.eq.slant z
    arrow.r.double w+x lt.eq.slant y + z
  )$
      ],
    )
  },
)

#pagebreak()
#text(size: 15pt, weight: "bold")[Step 3: Known Facts]
\
#diagram(

  edge-stroke: .07em,
  {

    node((0, 0), [$7 lt.eq.slant 3 + 9$])

    edge((0, 0), (0, 2), "-|>")
    node(
      (0, 2),
      [
        $forall w, x, y, z #h(0.2cm) (w
    lt.eq.slant y and x lt.eq.slant z
    arrow.r.double w+x lt.eq.slant y + z
  )$
      ],
    )

    edge((0, 2), (0.5, 4), "-|>")
    edge((0, 2), (-0.5, 4), "-|>")
    node(
      (0.5, 4),
      [
        $0 lt.eq.slant 3$
      ],
    )
    node(
      (-0.5, 4),
      [
        $7 lt.eq.slant 9$
      ],
    )
  },
)
#h(2cm)Hence Proved!

\
\
\
#underline(
  text(size: 17pt, weight: "bold")[Forward Chaining],
)
\
\
#text(size: 15pt, weight: "semibold")[Step 1: Known Facts]

#diagram(

  edge-stroke: .07em,
  {

    node(
      (0.5, 0),
      [
        $0 lt.eq.slant 3$
      ],
    )
    node(
      (-0.5, 0),
      [
        $7 lt.eq.slant 9$
      ],
    )
  },
)
\
\
#text(size: 15pt, weight: "semibold")[Step 2: Rules that satisfy the facts]
\
#diagram(

  edge-stroke: .07em,
  {

    node(
      (0, 2),
      [
        $forall w, x, y, z #h(0.2cm) (w
    lt.eq.slant y and x lt.eq.slant z
    arrow.r.double w+x lt.eq.slant y + z
  )$
      ],
    )

    edge((0, 2), (0.5, 0), "<|-")
    edge((0, 2), (-0.5, 0), "<|-")
    node(
      (0.5, 0),
      [
        $0 lt.eq.slant 3$
      ],
    )
    node(
      (-0.5, 0),
      [
        $7 lt.eq.slant 9$
      ],
    )
  },
)
\
\
#text(size: 15pt, weight: "semibold")[Step 3: Conlusion]
\
#diagram(

  edge-stroke: .07em,
  {

    node((0, 4), [$7 lt.eq.slant 3 + 9$])

    edge((0, 4), (0, 2), "<|-")
    node(
      (0, 2),
      [
        $forall w, x, y, z #h(0.2cm) (w
    lt.eq.slant y and x lt.eq.slant z
    arrow.r.double w+x lt.eq.slant y + z
  )$
      ],
    )

    edge((0, 2), (0.5, 0), "<|-")
    edge((0, 2), (-0.5, 0), "<|-")
    node(
      (0.5, 0),
      [
        $0 lt.eq.slant 3$
      ],
    )
    node(
      (-0.5, 0),
      [
        $7 lt.eq.slant 9$
      ],
    )
  },
)

#h(2cm)Hence Proved!
#pagebreak()
#problem[
  Generative AI (GenAI) is a subfield of Artificial Intelligence (AI). It focuses on algorithms that
  can generate new content, such as text, images, audio, and video, that resembles human-
  created content. Two such popular tools are: a) Microso ’s Copilot and b) OpenAI's ChatGPT.
  Use Copilot and ChatGPT to solve the following ques on. Check the correctness of the answers
  generated by these tools and do a comparative study, which one can make better results.

  #image("q3.png")
]

#text(
  size: 15pt,
  weight: "bold",
)[Generated text by two language model for the given problem]
#showybox(
  breakable: true,
  [ChatGPT],
  [
    To represent the given sentences in logical form, suitable for use with Generalized Modus Ponens (GMP), we'll break down each sentence into predicate logic step-by-step.

    == Symbols:
    - Let the predicate \( Mammal(x) \) denote "x is a mammal."
    - Let \( Horse(x) \) denote "x is a horse."
    - Let \( Cow(x) \) denote "x is a cow."
    - Let \( Pig(x) \) denote "x is a pig."
    - Let \( Offspring(x, y) \) denote "x is an offspring of y."
    - Let \( Parent(x, y) \) denote "x is a parent of y."
    - Let \( Bluebeard \) and \( Charlie \) be individual constants.

    == Logical Representations:

    **(a) Horses, cows, and pigs are mammals.**

    This sentence can be broken down into three individual statements:

    - \( \forall x \, (Horse(x) \rightarrow Mammal(x)) \)
    - \( \forall x \, (Cow(x) \rightarrow Mammal(x)) \)
    - \( \forall x \, (Pig(x) \rightarrow Mammal(x)) \)

    **(b) An offspring of a horse is a horse.**

    This can be written as:

    - \( \forall x \, \forall y \, (Offspring(x, y) \land Horse(y) \rightarrow Horse(x)) \)

    **(c) Bluebeard is a horse.**

    Since "Bluebeard" is an individual constant, we simply write:

    - \( Horse(Bluebeard) \)

    **(d) Bluebeard is Charlie’s parent.**

    This is written as:

    - \( Parent(Bluebeard, Charlie) \)

    **(e) Offspring and parent are inverse relations.**

    This indicates a bi-conditional relationship between offspring and parent:

    - \( \forall x \, \forall y \, (Offspring(x, y) \leftrightarrow Parent(y, x)) \)

    **(f) Every mammal has a parent.**

    This is represented as:

    - \( \forall x \, (Mammal(x) \rightarrow \exists y \, Parent(y, x)) \)

    ---

    == Final Summary:
    1. \( \forall x \, (Horse(x) \rightarrow Mammal(x)) \)
    2. \( \forall x \, (Cow(x) \rightarrow Mammal(x)) \)
    3. \( \forall x \, (Pig(x) \rightarrow Mammal(x)) \)
    4. \( \forall x \, \forall y \, (Offspring(x, y) \land Horse(y) \rightarrow Horse(x)) \)
    5. \( Horse(Bluebeard) \)
    6. \( Parent(Bluebeard, Charlie) \)
    7. \( \forall x \, \forall y \, (Offspring(x, y) \leftrightarrow Parent(y, x)) \)
    8. \( \forall x \, (Mammal(x) \rightarrow \exists y \, Parent(y, x)) \)

    This set of logical formulas can be used with Generalized Modus Ponens to infer new facts from the premises.
  ],
)

#pagebreak()

#showybox(
  breakable: true,
  [Copilot],
  [
    #text(
      size: 17pt,
      weight: "bold",
    )[
      1.Analyzing the Sentences and Representing Them Logically
    ]

    #text(
      size: 15pt,
      weight: "bold",
    )[
      Understanding the Problem\
    ]
    We need to express the given sentences in a logical form suitable for Generalized Modus Ponens. This involves using predicates, constants, and quantifiers.

    #text(
      size: 15pt,
      weight: "bold",
    )[
      Defining Predicates and Constants\
    ]
    **M(x):** x is a mammal
    **H(x):** x is a horse
    **C(x):** x is a cow
    **P(x):** x is a pig
    **O(x, y):** x is an offspring of y
    **B:** Bluebeard
    **Ch:** Charlie

    #text(
      size: 15pt,
      weight: "bold",
    )[
      Representing the Sentences\
    ]
    1. **a. Horses, cows, and pigs are mammals.**
      ∀x(H(x) → M(x))
      ∀x(C(x) → M(x))
      ∀x(P(x) → M(x))

    2. **b. An offspring of a horse is a horse.**
      ∀x∀y(H(y) ∧ O(x, y) → H(x))

    3. **c. Bluebeard is a horse.**
      H(B)

    4. **d. Bluebeard is Charlie's parent.**
      O(Ch, B)

    5. **e. Offspring and parent are inverse relations.**
      ∀x∀y(O(x, y) ↔ O(y, x))

    6. **f. Every mammal has a parent.**
      ∀x(M(x) → ∃yO(x, y))


    #text(
      size: 15pt,
      weight: "bold",
    )[
      Explanation\
    ]
    **Quantifiers:** ∀ (for all) and ∃ (there exists) are used to express generality and existence.
    **Predicates:** M, H, C, P, and O represent properties or relations.
    **Constants:** B and Ch represent specific individuals.
    **Implication (→):** This logical connective indicates that one statement implies another.
    **Biconditional (↔):** This logical connective indicates that two statements are equivalent.

    These logical representations can now be used with Generalized Modus Ponens to derive conclusions from the given premises.
  ],
)

#text(size: 15pt, weight: "bold")[Comparative Study]
\
The answer generated by ChatGPT provide a suitable logical representation for the given problem.

\
However in case of Copilot, while most of the representations are correct and often match
the answer generated by ChatGPT, its answer (Copilot) has few inconsistencies:

#pagebreak()
- Bluebeard is Charlie's parent: O(Ch, B)\
  This differs from ChatGPT's answer. It uses the offspring relation instead of the parent relation.
  While logically correct this is unlikely to be seen in human generated answers.
- Offspring and parent are inverse relations: ∀x∀y(O(x, y) ↔ O(y, x))\
  This is incorrect. It states that the offspring relation is symmetric, which is not true.\
Therefore only ChatGPT generated correct and consistent answer.
\
\
#text(size: 15pt, weight: "bold")[
  My approach after reviewing the generated answer would be:
]

1. Define predicates:
  - Mammal(x): x is a mammal
  - Horse(x): x is a horse
  - Cow(x): x is a cow
  - Pig(x): x is a pig
  - Offspring(x,y): x is an offspring of y
  - Parent(x,y): x is a parent of y
  - Bluebeard, Charlie: constants


#par(hanging-indent: 0.5cm)[
  2. Logical representations:
  \
  a. Horses, cows, and pigs are mammals:
  \
  #h(1cm)∀x(Horse(x) → Mammal(x))\
  #h(1cm)∀x(Cow(x) → Mammal(x))\
  #h(1cm)∀x(Pig(x) → Mammal(x))
  \
  b. An offspring of a horse is a horse:
  \
  #h(1cm)∀x∀y(Offspring(x,y) ∧ Horse(y) → Horse(x))
  \
  c. Bluebeard is a horse:
  \
  #h(1cm)Horse(Bluebeard)
  \
  d. Bluebeard is Charlie's parent:
  \
  #h(1cm)Parent(Bluebeard, Charlie)
  \
  e. Offspring and parent are inverse relations:
  \
  #h(1cm)∀x∀y(Offspring(x,y) ↔ Parent(y,x))
  \
  f. Every mammal has a parent:
  \
  #h(1cm)∀x∃y (Mammal(x) → Parent(y,x))
]
