#import "lib.typ": *


#show: doc => assignment(
  title: "Digital Assignment - I",
  course: "Software Engineering",
  // Se apenas um autor colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "2 February 2025",
  doc,
)

= Question

#cblock[
  Suppose a travel agency needs a software for automating its bookkeeping activities. The set of activities to be automated are rather simple and are at present being carried out manually. The travel agency has indicated that it is unsure about the type of user interface which would be suitable for its employees and its customers. Would it be proper for a development team to use the spiral model for developing this software?\
\
*[2 marks]*
]

== Answer

No, it would not be proper for the development team to use the spiral model for developing this software. While the spiral model is a robust and comprehensive approach, it is generally more suitable for large, complex projects with unclear or high-risk requirements.

In this scenario, the requirements are relatively simple and well-defined (bookkeeping activities). The main uncertainty lies in the user interface, which can be addressed through simpler iterative models like the iterative or incremental model with rapid prototyping.

These models allow for developing a basic version of the software quickly and iteratively refining it based on user feedback, especially regarding the user interface. This approach is more efficient and cost-effective for this specific situation compared to the spiral model's comprehensive and potentially time-consuming cycles.

= Question

#cblock[
  Galaxy Inc. undertook the development of a satellite-based communication between mobile handsets that can be anywhere on the earth. In contrast to the traditional cell phones, by using a satellite-based mobile phone a call can be established as long as both the source and destination phones are in the coverage areas of some base stations. The system would function through about six dozens of satellites orbiting the earth. The satellites would directly pick up the signals from a handset and beam signal to the destination handset. Since the foot prints of the revolving satellites would cover the entire earth, communication between any two points on the earth, even between remote places such as those in the Arctic ocean and Antarctica, would also be possible. However, the risks in the project are many, including determining how the calls among the satellites can be handed-off when they are themselves revolving at a very high speed. In the absence of any published material and availability of staff with experience in development of similar products, many of the risks cannot be identified at the start of the project and are likely to crop up as the project progresses. The software would require several million lines of code to be written. What software model Galaxy Inc. should use?\
\
*[2 marks]*
]

== Answer
Galaxy Inc. should use the spiral model for the development of its satellite-based communication system. The spiral model is particularly suitable for this project due to its flexibility and high risk-handling capabilities. Given the complexity and numerous risks associated with developing a satellite-based communication system, the spiral model's iterative approach allows for comprehensive risk analysis at each step. This ensures that potential issues, such as the hand-off of calls among rapidly moving satellites, can be identified and mitigated early in the development process. Additionally, the lack of published material and experienced staff necessitates a methodical and adaptive approach, which the spiral model provides by incorporating feedback and adjustments at every iteration.\

= Question
#cblock[
  Assume that the size of an organic type software product has been estimated to be 32,000 lines of source code. Assume that the average salary of a software developer is Rs. 15,000 per month. Determine the effort required to develop the software product, the nominal development time, and the cost to develop the product.\
\
*[3 marks]*
]

== Answer
To estimate the effort, development time, and cost for the software product, we can use the COCOMO (Constructive Cost Model) model. Since the problem specifies an "organic" type software product, we will use the organic mode equations.
\
\
*1. Effort $(E)$:*\
\The formula for effort: $E = 2.4 * (K L O C)^1.05$\
\
$E = 2.4 * (32)^1.05$\
$E ≈ 91.33$ person-months\
\
*2. Nominal Development Time $(T)$:*\
The formula for development time in the organic mode is: $T = 2.5 * (E)^0.38$\
\
$ T = 2.5 * (91.33)^0.38$\
$ T ≈ 13.9$ months\
\
\
*3. Cost $(C)$:*\
To calculate the cost, we need to know the average salary of a software developer per month, which is given as Rs. 15,000.\
\
$C = E * A v g$\
$C = 91.33 * 15,000$\
$C = 13,69,966.6$\
\
*Results:*
\
Effort: Approximately $91$ person-months.\
Nominal Development Time: Approximately $14$ months.\
Cost: Approximately Rs. $13,69,966.6$.\

= Question
#cblock[
Determine the function point measure of the size of the following supermarket software. A supermarket needs to develop the following software to encourage regular customers. For this, the customer needs to supply his/her residence address, telephone number, and the driving license number. Each customer who registers for this scheme is assigned a unique customer number (CN) by the computer. Based on the generated CN, a clerk manually prepares a customer identity card after getting the market manager’s signature on it. A customer can present his customer identity card to the check out staff when he makes any purchase. In this case, the value of his purchase is credited against his CN. At the end of each year, the supermarket intends to award surprise gifts to 10 customers who make the highest total purchase over the year. Also, it intends to award a 22 caret gold coin to every customer whose purchase exceeded Rs. 10,000. The entries against the CN are reset on the last day of every year after the prize winners’ lists are generated. Assume that various project characteristics determining the complexity of software development to be average.\
\
*[4 marks]*
]

== Answer

*In given scenario:*\
1. External Inputs (EIs) $ = 3$
    - Customer Registration
    - Purchase Entry
    - Year‐End Reset
2. External Outputs (EOs) $ = 2$\
    - Customer Information for Card Preparation\
    - Annual Award Reports\
3. External Inquiries (EQs) $ = 2$\
    - Retrieve Customer Details for ID Card\
    - Verify Customer at Checkout\
4. Data Functions (ILFs) $ = 2$\
    - Customer Master File\
    - Purchase (Transaction) File\
5. External Interface Files(EIFs) $ = 0$\
\
*Assign Standard Average Weights*\
- EI (External Input): 10 
- EO (External Output): 7 
- EQ (External Inquiry): 4 
- ILF (Internal Logical File): 5 

*Now, calculate the unadjusted function points (UFP):*
- EIs:  $3 × 10 = 30 F P$
- EOs:  $2 × 7 = 14 F P$
- EQs:  $2 × 4 = 8 F P$
- ILFs: $2 × 5 = 10 F P$
*Total UFP* = $30 + 14 + 8 + 10 = 62 F P$
\
\
*Apply the Value Adjustment Factor (VAF)*\
- Total Degree of Influcence $ = 14 × 3 = 42$
- VAF $ = 0.65+(0.01×42) = 1.07$

*Now, calculate the adjusted function points (AFP):*\
- AFP $ =U F P × V A F$
- AFP $ =62×1.07≈66.34$

*Rounding off, we get approximately 66 Function Points.*
