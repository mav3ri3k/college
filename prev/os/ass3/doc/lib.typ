#let report(
  title: "Typst IFSC",
  subtitle: none,
  authors: ("Gabriel Luiz Espindola Pedro",),
  date: none,
  doc,
) = {
  // Define metadados do documento
  set document(title: title, author: authors)
  set page(
    numbering: "1",
    paper: "a4",
    margin: (top: 2cm, bottom: 2cm, left: 1cm, right: 1cm),
  )
  // TODO: verificar se há necessidade de colocar espaçamento de 1.5
  set par(
    first-line-indent: 1.5cm,
    justify: true,
    leading: 0.65em,
    linebreaks: "optimized",
  )
  set heading(numbering: "1.")
  set math.equation(numbering: "(1)")

  align(center)[
    #image("title.png", width: 35em)

    #text(23pt, weight: "semibold", fill: rgb(55, 63, 104))[
      B.Tech. Winter Semester 2023-24\
      School Of Computer Science and Engineering\
      (SCOPE)\
    ]

    #align(horizon)[
      #text(40pt, title, weight: "bold")\
      #v(1em)
      #text(27pt, subtitle, weight: "semibold")
    ]

    #text(
      27pt,
      list(..authors, marker: "", body-indent: 0pt),
      weight: "semibold",
    )
    #text(20pt, date)

  ]
  pagebreak()

  /*
  show outline.entry.where(level: 1): it => {
  strong(it)
  }

  // TODO: Verificar maneira melhor de alterar espaçamento entre titulo e corpo
  outline(title: [Sumário #v(1em)], indent: 2em)

  pagebreak()
  */
  doc
}
