#import "lib.typ": *
#import "@preview/fletcher:0.5.4":*

#show: doc => assignment(
  title: "Notes",
  course: "Software Engineering",
  // Se apenas um author colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "CAT - II",
  doc,
)

= Module:2 Introduction To Software Project Management
== Risk Management
== RMMM Plan
== CASE TOOLS

= Module:3 Modelling Requirements

== System Modeling
== Requirements Specification and Requirement Validation
=== Requirements Elicitation techniques
=== Requirements management in Agile.

= Module:4 Software Design
== Design concepts and principles
Converting Software Requirement Specification (_SRS_) document to a design document.\
\
*Parts of design process:*
1. Interface Design
2. Architecture Design
3. Data/Class Design
4. Component Level Design

*Mapping Diagram to Design Model:*
#table(
    columns: (auto, auto, auto, auto),
    table.header([Interface Design], [Architecture Design], [Data/Class Design], [Component Design]),
    [
        - Use Case
        - Activity
        - Data Flow
        - State
        - Sequence 
    ],
    [
      - Class 
    - Analysis   
    ],
    [
        - Data-Flow
        - Class
        - Anaylysis
    ],
    [
        - Class
        - Data Flow
        - Sequence Diagram
    ],

)
#figure(
    image("./diag_map.png"),
    caption: [
        Diagram to Design Model mapping
    ]
)
== Abstraction
== Refinement
== Modularity Cohesion coupling
== Architectural design
== Detailed Design Transaction Transformation
== Refactoring of designs
== Object oriented Design User-Interface Design

= Module:5 Validation And Verification
== Strategic Approach to Software Testing
=== Testing Fundamentals Test Plan
=== Test Design
=== Test Execution, Reviews
=== Inspection and Auditing

== Regression Testing
== Mutation Testing
== Object oriented testing

= Diagram
#figure(
    image("./diagrams.png"),
    caption: [Types of Diagrams]
)
== DFD (1, 2)
== Use Case
== Sequence
== Class
== Activity
== ER
== State Transition
