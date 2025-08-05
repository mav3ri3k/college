#import "lib.typ": *
#import "@preview/touying:0.6.1":*
#import "@preview/cetz:0.3.4"
#import "@preview/fletcher:0.5.8":*


#show: doc => assignment(
  title: "Marching Cubes Algorithm",
  course: "Graphics Design",
  author: "Apratim Mishra",
  date: "07 June, 2025",
  doc,
)

#let tsize = 20pt

#text-card(tsize)[
= Problem
- Medical imaging techniques like MRI and CT scan only produce 2d slices of images
- Using these 2D image stacks, we want to visualize them in 3D
]
#pagebreak()

#set text(size: 15pt)
#block(
  breakable: false,
  width: 100%,
)[
  #align(center, [
    #image(width: 39em, "./mri.jpg")

    #block(
      breakable: false,
      radius: radius,
      fill: surface0,
      inset: (x: 1em, y: 1em),
    )[2D image slices received from MRI scan]

    ]
  )
]
#pagebreak()

#block(
  breakable: false,
  width: 100%,
)[
  #align(center, [
    #image(width: 34em, "./ct.jpg")

    #block(
      breakable: false,
      radius: radius,
      fill: surface0,
      inset: (x: 1em, y: 1em),
    )[2D image slices received from CT scan]

    ]
  )
]
#pagebreak()

#text-card(tsize)[
= Solution
Algorithm developed by _William E. Lorensen_ and _Harvey E. Cline_
published in 1987 SIGGRAPH proceedings
]
#set text(size: 10pt)
#align(center,
grid(
  columns: (auto, auto),
  column-gutter: 5em,
  image(width: 43em, "./paper.png"),
  diagram(
  	node-stroke: 1pt,
    node-corner-radius: 4pt,
    node-fill: surface0,
    node((0, 0), [Data Acquisition\ CT MRI]),
    edge("-|>"),
    node((0, 1), [Image Processing]),
    edge("-|>"),
    node((0, 2), [Model Creation]),
    edge("-|>"),
    node((0, 3), [View Operations]),
    edge("-|>"),
    node((0, 4), [Display]),
)

)
)
#set text(size: 15pt)

#pagebreak()

#align(center + horizon,
grid(columns: (auto, 20em),
align: center + horizon,
column-gutter: 2em,
cetz.canvas({
import cetz.draw: *
scale(150%)
grid((0, 0), (1, 1), stroke: 0.5pt)
hobby((-0.5, 0.4),(0.3, 0.3), (0.5, -0.4), omega: 0, stroke: blue)

grid((2, 0), (3, 1), stroke: 0.5pt)
hobby((1.5, 0.4),(2.3, 0.6), (3.4, 0.4), omega: 0, stroke: blue)

grid((4, 0), (5, 1), stroke: 0.5pt)
hobby((3.6, 0.4),(4.5, 0.6), (4.5, 1.5), omega: 0, stroke: blue)

}),
card("filled")[Object drawn on 2d grid],
cetz.canvas({
import cetz.draw: *
scale(150%)
grid((0, 0), (1, 1), stroke: 0.5pt)
hobby((-0.5, 0.4),(0.3, 0.3), (0.5, -0.4), omega: 0, stroke: blue)
circle((0, 0), radius: 2pt, fill: red)

grid((2, 0), (3, 1), stroke: 0.5pt)
hobby((1.5, 0.4),(2.3, 0.6), (3.4, 0.4), omega: 0, stroke: blue)
circle((2, 0), radius: 2pt, fill: red)
circle((3, 0), radius: 2pt, fill: red)

grid((4, 0), (5, 1), stroke: 0.5pt)
hobby((3.6, 0.4),(4.5, 0.6), (4.5, 1.5), omega: 0, stroke: blue)
circle((4, 0), radius: 2pt, fill: red)
circle((5, 0), radius: 2pt, fill: red)
circle((5, 1), radius: 2pt, fill: red)

}),

card("filled")[Points inside the object marked in #text(fill: red)[red]],
cetz.canvas({
import cetz.draw: *
scale(150%)
grid((0, 0), (1, 1), stroke: 0.5pt)
hobby((-0.5, 0.4),(0.3, 0.3), (0.5, -0.4), omega: 0, stroke: blue)
circle((0, 0), radius: 2pt, fill: red)
circle((0, 0.5), radius: 2pt, fill: black)
circle((0.5, 0), radius: 2pt, fill: black)

grid((2, 0), (3, 1), stroke: 0.5pt)
hobby((1.5, 0.4),(2.3, 0.6), (3.4, 0.4), omega: 0, stroke: blue)
circle((2, 0), radius: 2pt, fill: red)
circle((3, 0), radius: 2pt, fill: red)
circle((2, 0.5), radius: 2pt, fill: black)
circle((3, 0.5), radius: 2pt, fill: black)

grid((4, 0), (5, 1), stroke: 0.5pt)
hobby((3.6, 0.4),(4.5, 0.6), (4.5, 1.5), omega: 0, stroke: blue)
circle((4, 0), radius: 2pt, fill: red)
circle((5, 0), radius: 2pt, fill: red)
circle((5, 1), radius: 2pt, fill: red)
circle((4, 0.5), radius: 2pt, fill: black)
circle((4.5, 1), radius: 2pt, fill: black)

}),
card("filled")[Middle points activated due to red points marked in *black*],
cetz.canvas({
import cetz.draw: *
scale(150%)
grid((0, 0), (1, 1), stroke: 0.5pt)
hobby((-0.5, 0.4),(0.3, 0.3), (0.5, -0.4), omega: 0, stroke: blue)
circle((0, 0), radius: 2pt, fill: red)
circle((0, 0.5), radius: 2pt, fill: black)
circle((0.5, 0), radius: 2pt, fill: black)
line((0, 0.5), (0.5, 0), stroke: 2pt)

grid((2, 0), (3, 1), stroke: 0.5pt)
hobby((1.5, 0.4),(2.3, 0.6), (3.4, 0.4), omega: 0, stroke: blue)
circle((2, 0), radius: 2pt, fill: red)
circle((3, 0), radius: 2pt, fill: red)
circle((2, 0.5), radius: 2pt, fill: black)
circle((3, 0.5), radius: 2pt, fill: black)
line((2, 0.5), (3, 0.5), stroke: 2pt)

grid((4, 0), (5, 1), stroke: 0.5pt)
hobby((3.6, 0.4),(4.5, 0.6), (4.5, 1.5), omega: 0, stroke: blue)
circle((4, 0), radius: 2pt, fill: red)
circle((5, 0), radius: 2pt, fill: red)
circle((5, 1), radius: 2pt, fill: red)
circle((4, 0.5), radius: 2pt, fill: black)
circle((4.5, 1), radius: 2pt, fill: black)
line((4, 0.5), (4.5, 1), stroke: 2pt)

}),
card("filled")[Join the activated points],
)
)

#pagebreak()

#figure-card(
[
*Marching Squares in 2D*\
#align(left,
[
_1:_ Object traced on squares in #text(fill: blue)[blue]\
_2:_ Points inside the object in #text(fill:red)[red], points on boundary in *black*\ 
_3:_ Water tight traced mesh 
]
)
],
grid(
  columns: (auto, auto, auto),
  column-gutter: 1em,

cetz.canvas({
import cetz.draw: *
grid((-3, 4), (4, -3), stroke: 0.5pt)
hobby((-2.5, 0.5), (-2, 2), (2, 2), (2, -1.5), (1, 1), (-1, 0.2),(-2.5, 0.5), omega: 0, stroke: blue)
}),

cetz.canvas({
import cetz.draw: *
grid((-3, 4), (4, -3), stroke: 0.5pt)
let inside = (
    (-2,  1), (-2,  2), (-1,  2), (-1,  1),
     (0,  1), ( 0,  2),
     (1,  2), ( 1,  1),
     (2,  2), ( 2,  1),
     (3,  1), ( 2,  0), ( 3,  0),
     (3, -1), ( 2, -1),
  )
    for x in range(-3, 5) {
    for y in range(-3, 5) {
      let p = (x, y)
      if inside.contains(p) { 
        circle(p, radius: 0.1cm, fill: red, stroke: none)
      }
    }

    
  }

hobby((-2.5, 0.5), (-2, 2), (2, 2), (2, -1.5), (1, 1), (-1, 0.2),(-2.5, 0.5), omega: 0, stroke: blue)

  let boundary = (
    (-2,  2.5),
    (-1,  2.5),
    (0,  2.5),
    (1,  2.5),
    (2,  2.5),
    (2.5,  2),
    (3,  1.5),
    (3.5,  1),
    (3.5,  0),
    (3.5,  -1),
    (3,  -1.5),
    (2,  -1.5),
    (1.5,  -1),
    (1.5,  0),
    (1,  0.5),
    (0,  0.5),
    (-1,  0.5),
    (-2,  0.5),
    (-2.5,  1),
    (-2.5,  2),
  )

  for p in boundary {
    circle(p, radius: 0.1cm, fill: rgb(1, 0, 1), stroke: none)
  }
}),

cetz.canvas({
import cetz.draw: *
grid((-3, 4), (4, -3), stroke: 0.5pt)

hobby((-2.5, 0.5), (-2, 2), (2, 2), (2, -1.5), (1, 1), (-1, 0.2),(-2.5, 0.5), omega: 0, stroke: blue)

  let boundary = (
    (-2,  2.5),
    (-1,  2.5),
    (0,  2.5),
    (1,  2.5),
    (2,  2.5),
    (2.5,  2),
    (3,  1.5),
    (3.5,  1),
    (3.5,  0),
    (3.5,  -1),
    (3,  -1.5),
    (2,  -1.5),
    (1.5,  -1),
    (1.5,  0),
    (1,  0.5),
    (0,  0.5),
    (-1,  0.5),
    (-2,  0.5),
    (-2.5,  1),
    (-2.5,  2),
  )

  for p in boundary {
    circle(p, radius: 0.1cm, fill: rgb(1, 0, 1), stroke: none)
  }
  line(
    (-2,  2.5),
    (-1,  2.5),
    (0,  2.5),
    (1,  2.5),
    (2,  2.5),
    (2.5,  2),
    (3,  1.5),
    (3.5,  1),
    (3.5,  0),
    (3.5,  -1),
    (3,  -1.5),
    (2,  -1.5),
    (1.5,  -1),
    (1.5,  0),
    (1,  0.5),
    (0,  0.5),
    (-1,  0.5),
    (-2,  0.5),
    (-2.5,  1),
    (-2.5,  2),
    (-2, 2.5),
    )
})

)
)
#pagebreak()

#grid(
  columns: (auto, auto),
  column-gutter: 2em,
cetz.canvas({
import cetz.draw: *
scale(170%)
grid((-3, 4), (4, -3), stroke: 0.5pt)

hobby((-2.5, 0.5), (-2, 2), (2, 2), (2, -1.5), (1, 1), (-1, 0.2),(-2.5, 0.5), omega: 0, stroke: blue, close: true, fill: blue)

  let boundary = (
    (-2,  2.1),
    (-1,  2.5),
    (0,  2.55),
    (1,  2.45),
    (2,  2.1),
    (2.1,  2),
    (3,  1.2),
    (3.2,  1),
    (3.55,  0),
    (3.25,  -1),
    (3,  -1.35),
    (2,  -1.5),
    (1.5,  -1),
    (1.5,  0),
    (1,  1),
    (0,  0.9),
    (-1,  0.2),
    (-2,  0.1),
    (-2.65,  1),
    (-2.1,  2),
  )

  for p in boundary {
    circle(p, radius: 0.1cm, fill: rgb(1, 0, 1), stroke: none)
  }

  line(
    ..boundary,
    stroke: 3pt
  )

}),
card("filled")[
*Optimisation*\

After the last step, move the points closer to object boundary,
by moving it along its edge axis without going out of the edge boundary. 
]
)

#pagebreak()

#figure-card([
*Marching Cube in 3D*\
#align(left, [
_1:_ #text(fill: blue)[Object] traced in cube\
_2:_ Mark mid points to make shape around the object, shown in #text(fill: red)[red]\
_3:_ Move the points along the respective edge axis for optimisation
])],
grid(
  columns: (auto, auto, auto),
  column-gutter: 1em,

cetz.canvas({
import cetz.draw: *
grid((-3, 4), (4, -3), stroke: 0.5pt)
on-layer(1,
  line(
    (-2, 2),
    (2, 2),
    (2, -2),
    (-2, -2),
    (-2, 2),
  )
)
line(
  (-1, 3),
  (3, 3),
  (3, -1),
  (-1, -1),
  (-1, 3),
)

line((-2, 2), (-1, 3))
line((2, 2), (3, 3))
line((2, -2), (3, -1))
line((-2, -2), (-1, -1))

line((-2, -1), (-1, 0.5), (3, 0.7), (2, -1.7), close: true, fill: blue)
}),

cetz.canvas({
import cetz.draw: *
grid((-3, 4), (4, -3), stroke: 0.5pt)
on-layer(1,
  line(
    (-2, 2),
    (2, 2),
    (2, -2),
    (-2, -2),
    (-2, 2),
  )
)
line(
  (-1, 3),
  (3, 3),
  (3, -1),
  (-1, -1),
  (-1, 3),
)
line((-2, 2), (-1, 3))
line((2, 2), (3, 3))
line((2, -2), (3, -1))
line((-2, -2), (-1, -1))

on-layer(1,{
circle((-2, 0), fill: red, radius: 3pt)
circle((-1, 1), fill: red, radius: 3pt)
circle((3, 1), fill: red, radius: 3pt)
circle((2, 0), fill: red, radius: 3pt)
})

line((-2, -1), (-1, 0.5), (3, 0.7), (2, -1.7), close: true, fill: blue)
line((-2, 0), (-1, 1), (3, 1), (2, 0), close: true, fill: red)
})
,
cetz.canvas({
import cetz.draw: *
grid((-3, 4), (4, -3), stroke: 0.5pt)
on-layer(1,
  line(
    (-2, 2),
    (2, 2),
    (2, -2),
    (-2, -2),
    (-2, 2),
  )
)
line(
  (-1, 3),
  (3, 3),
  (3, -1),
  (-1, -1),
  (-1, 3),
)

line((-2, 2), (-1, 3))
line((2, 2), (3, 3))
line((2, -2), (3, -1))
line((-2, -2), (-1, -1))

on-layer(1,{
  circle((-2, -0.9), fill: red, radius: 3pt)
  circle((-1, 0.45), fill: red, radius: 3pt)
  circle((3, 0.65), fill: red, radius: 3pt)
  circle((2, -1.8), fill: red, radius: 3pt)
})

line((-2, -1), (-1, 0.5), (3, 0.7), (2, -1.7), close: true, fill: blue)
line((-2, -0.9), (-1, 0.45), (3, 0.65), (2, -1.8), close: true, fill: red)

})
)
)

#pagebreak()
#align(horizon,
grid(
  columns: (auto, auto),
  column-gutter: 1em,

image-card("All 15 possible cases", "./MarchingCubesCases.png"),
align(top,
card("filled")[
  - Since each vertex can either be #text(fill: green)[outside] or #text(fill: red)[inside], there are technically $2^8 = 256$ possible configurations, but many of these are equivalent to one another.\
  \
  - There are only $15$ unique cases, shown here.\
  \
  - This allows for easy triangle generation using lookup table for each case]
)
))
#pagebreak()

#set text(size: 10pt)
#align(center + horizon,
grid(
  align: center + horizon,
  columns: (auto, auto, auto),
  column-gutter: 2em,
  row-gutter: 2em,
  image("./1.png"),
  image("./2.png"),
  image("./3.png"),
  image("./4.png"),
  image("./5.png"),
  image("./6.png"),
)
)
#set text(size: 15pt)
#pagebreak()

#text-card(tsize)[
= Implementation Details
\
- *Data Structures:* Efficient storage of vertex and edge information is crucial.
- *Optimization:* Techniques like edge and vertex caching can improve performance.
- *Parallelization:* The algorithm is well-suited for parallel processing due to the independence of cube evaluations.
]

#pagebreak()

#text-card(tsize)[
= Applications
\
- *Medical Imaging:* Visualization of anatomical structures from CT and MRI scans.
- *Scientific Visualization:* Representation of scalar fields in physics and engineering.
- *Computer Graphics:* Modeling complex surfaces and terrains.
]
#grid(
  columns: (auto, 11em),
  column-gutter: 1em,
  align: center + bottom,

image(width: 16em, "./lidar.jpg"),
card("filled")[*Lidar Point Cloud*]
)
#pagebreak()

#text-card(tsize)[
= Advantages
\
- *High Resolution:* Produces detailed and accurate 3D surfaces.
- *Efficiency:* Capable of processing large datasets effectively.
- *Versatility:* Applicable to various fields requiring 3D visualization.
]

#grid(
  columns: (auto, auto),
  column-gutter: 2em,
  align: center + horizon,

  image("./up1.png"),
  image("./up2.png"),
)

#pagebreak()

#text-card(tsize)[
= Future Retrospective 
]
#align(center,
image(width: 11em, "./model.svg")
)

#block(
  breakable: false,
  width: 100%,
)[
  #align(center, [
    #image(width: 35em, "./nerf.png")

    #block(
      breakable: false,
      radius: radius,
      fill: surface0,
      inset: (x: 1em, y: 1em),
      )[*NeRF*: Representing Scenes as Neural Radiance Fields for View Synthesis]
    ]
  )
]
#pagebreak()
#align(center + horizon, text(font:"Vollkorn", size: 70pt, weight: "bold")[THANK YOU])
