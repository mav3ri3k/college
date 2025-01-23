#import "lib.typ": *
#import "@preview/fletcher:0.5.4": *

#show: doc => assignment(
  title: "Digital Assignment - I",
  course: "Embedded Systems",
  // Se apenas um autor colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "22 January 2024",
  doc,
)

= Question

#cblock[
  Prepare a detailed analysis on the Automatic Garden Watering System in the following aspects (Digital or Hand written).
  - Purpose and significance of the project.
  - Requirement Analysis,
  - Functional Diagram,
  - Block Diagram, UML Diagrams ( Use Case Diagram and/or/Sequence Diagram)
]

== Answer
*Purpose*\
The primary purpose of an Automatic Garden Watering System is to automate the process of watering plants in a garden or any designated area. It aims to deliver the right amount of water to plants at the right time, without manual intervention.
\
\
*Significance*
- Water Conservation: Optimizes water usage, reducing waste.
- Time Savings: Automates a time-consuming chore.
- Convenience: Provides consistent watering, even when the user is away.
- Plant Health: Promotes healthier plant growth through consistent moisture.

* Requirement Analysis*\
Functional Requirements:
- Soil Moisture Sensing: Detect soil moisture levels.
- Automated Watering: Trigger water flow when moisture is low.
- Adjustable Thresholds: Allow users to set moisture trigger points.
- Programmable Schedules: Enable watering at set times/intervals.
- Manual Override: Provide a manual on/off switch.

Non-Functional Requirements:
- Reliability: Consistent operation.
- Durability: Weather-resistant components.
- Power Efficiency: Minimize energy consumption.
- Ease of Use: Simple user interface.
- Safety: No electrical or water hazards.

#pagebreak()
*Functional Diagram*\
#block()[
  #align(center)[
    #diagram(
      node-stroke: 1pt,
      node((0, 0), [User Interface], corner-radius: 0.5em),
      node((1, 0), [Control Unit], corner-radius: 0.5em),
      node((2, 0), [Watering System], corner-radius: 0.5em),
      node((1, 1), [Sensor], corner-radius: 0.5em),

      edge((0, 0), (1, 0), "-|>"),
      edge((1, 0), (2, 0), "-|>"),
      edge((1, 0), (1, 1), "-|>"),
      edge((1, 1), (0, 1), (0, 0), "-|>"),
      edge((1, 1), (2, 1), (2, 0), "-|>"),
    )
  ]
]

\
\
\
*Block Diagram*\
#text(10pt)[
  #block()[
    #align(center)[
      #diagram(
        node-stroke: 1pt,
        node((-1, 0), [Power Supply], corner-radius: 0.5em),
        node((0, 1), [Soil Moisture\ Sensor], corner-radius: 0.5em),
        node((0, 2), [Rain Sensor], corner-radius: 0.5em),
        node((0, 3), [Temperature\ Sensor], corner-radius: 0.5em),

        node((1, 0), [Microcontroller\ (Control Unit)], corner-radius: 0.5em),

        node((2, 0), [Relay/Driver], corner-radius: 0.5em),
        node((2, 1), [Solenoid Valve\ (Water Control)], corner-radius: 0.5em),
        node((2, 2), [Water Source], corner-radius: 0.5em),
        node(
          (2, 3),
          [Irrigation System],
          corner-radius: 0.5em,
          extrude: (0, 3),
        ),

        edge((-1, 0), (1, 0), "-|>"),
        edge((-1, 0), (-1, 1), (0, 1), "-|>"),
        edge((-1, 0), (-1, 2), (0, 2), "-|>"),
        edge((-1, 0), (-1, 3), (0, 3), "-|>"),

        edge((0, 1), (1, 1), (1, 0), "-|>"),
        edge((0, 2), (1, 2), (1, 0), "-|>"),
        edge((0, 3), (1, 3), (1, 0), "-|>"),

        edge((1, 0), (2, 0), "-|>"),
        edge((2, 0), (2, 1), "-|>"),
        edge((2, 1), (2, 2), "-|>"),
        edge((2, 2), (2, 3), "-|>"),
      )
    ]
  ]
]

#pagebreak()

= Question

#cblock[
  Describe the special purpose register used for interrupt configuration. Assume that two switches are connected to pin p3.2 p3.3 . write a program to monitor the switch and perform the following using external hardware interrupt. if SW1=0, FLASH port0 with (FF to 00) if SW2 =0 , FLASH port2 with (55 to AA)
]

== Answer
#image("Embedass1_1.jpg", height: 77%)
#image("Embedass1_2.jpg")

= Question

#cblock[
  The 8051 microcontroller is configured for serial communication at a baud rate of 9600 using a crystal frequency of 11.0592 MHz. The UART is set in Mode 1 (8-bit data, variable baud rate), and Timer 1 is used in Mode 2 (8-bit auto-reload mode) to generate the baud rate. What value should be loaded into the Timer 1 TH1 register to achieve the desired baud rate?
]

== Answer
#image("Embedass1_3.jpg", height: 75%)

= Question

#cblock[
  Draw the following in A3.
  1. architecture of 8051 microcontroller
  2. PIn diagram
  3. RAM (4 register banks with its address, bit addressable field)
  4. SPR with address
  5. TCON
  6. TMOD
  7. SCON
  8. PCON
  9. DPTR
]

== Answer
#image("Embedass1_4.jpg", height: 65%)
