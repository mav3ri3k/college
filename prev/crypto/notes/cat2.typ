#import "lib.typ": *
#import "@preview/fletcher:0.5.4":*

#show: doc => assignment(
  title: "Notes",
  course: "Cryptography and Network Security",
  // Se apenas um author colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "CAT - II",
  doc,
)


= Module 3: Asymmetric Encryption Algorithm and Key Exchange
#diagram(
    node-stroke: 1pt,
    node((0, 0), [Message]),
    node((1, 0), [E]),
    node((2, 0), [E]),
    node((3, 0), [D]),
    node((4, 0), [D]),
    node((5, 0), [Message]),

    node((1, 1), [R$K$#sub[pub]]),
    node((2, 1), [S$K$#sub[pri]]),
    node((3, 1), [S$K$#sub[pub]]),
    node((4, 1), [R$K$#sub[pri]]),

    edge((1,1), (1, 0), "-|>"),
    edge((2,1), (2, 0), "-|>"),
    edge((3,1), (3, 0), "-|>"),
    edge((4,1), (4, 0), "-|>"),

    edge((0, 0), (1, 0), "-|>"),
    edge((1, 0), (2, 0), "-|>"),
    edge((2, 0), (3, 0), "-|>", $T r a n s i t$),
    edge((3, 0), (4, 0), "-|>"),
    edge((4, 0), (5, 0), "-|>"),
)

== Principles
#table(
fill: (x, y) => {
    if y == 0 {bg}
},
  columns: (auto, auto, auto, auto),
    [Algorithm],
    [Encryption/Decryption],
    [Digital Signature],
    [Key Exchange],
    [RSA], [Yes], [Yes], [Yes],
    [Elliptic Curve], [Yes], [Yes], [Yes],
    [Diffie-Hellman], [No], [No], [Yes],
)
== *RSA*
=== Steps
1. Choose two large primes:
$
P, Q\
N = P*Q
$

2. Choose public and private key:
$
K#sub[pub] &#sym.bar.v K#sub[pub] "is not factor of" #sym.phi.alt (N)\
K#sub[pri] &#sym.bar.v (K#sub[pri] * K#sub[pri]) "mod" #sym.phi.alt (N) = 1
$ 

3. Encrypt:
$
    C T = P T ^(K#sub[pub]) "mod" N
$

4. Decrypt:
$
    P T = C T ^ (K#sub[pri]) "mod" D
$

== *ElGamal*

1. Choose public numbers such that:
    - $#sym.alpha, q$ are prime
    - $#sym.alpha$ is primitive root of $q$
$
    #sym.alpha, q&\
$

2. A: Compute
    - Private Key: $X#sub[A]$
    - Public Key : ${q, #sym.alpha, Y#sub[A]}$
$
    X#sub[A] &| X#sub[A] #sym.in (1, q-1)\ 
    Y#sub[A] &= #sym.alpha#super[X#sub[A]] "mod" q\ 
$ 

3. B
    - Message: $M | M #sym.in [1, q-1]$
    - Random : $k | k #sym.in [1, q-1]$

4. Encrypt $(C#sub[1], C#sub[2])$:
$
    C#sub[1] &= #sym.alpha^k "mod" q\
    C#sub[2] &= K M "mod" q
$

5. A: Decrypt
$
    K &= C#sub[1]^(X#sub[A]) "mod" q\
    M &= C#sub[2]K^(-1) "mod" q\
$

#cblock[
    If a message must be broken up into blocks and sent as a sequence of ­ encrypted
blocks, a unique value of k should be used for each block. If k is used for more than
one block, knowledge of one block M1 of the message enables the user to compute
other blocks as follows. Let
]

== *Elliptic Curve cryptography*
== Homomorphic Encryption and Secret Sharing
== Key distribution and Key exchange protocols
== *Diffie-Hellman Key Exchange*
1. Choose public numbers such that:
    - $g$ is primitive root of $n$
    - $g, n$ are primes
$
    g, n &\
$

2. Choose private numbers:
$
    x#sub[A] &#sym.bar.v x < n\
    y#sub[B] &#sym.bar.v y < n\
$

3. New public values:

$
    A &= g^x "mod" n\
    B &= g^y "mod" n\
$ 

4. Generate Keys User side:
$
    K#sub[A] &= B^x "mod" n\
    K#sub[B] &= A^y "mod" n\
    \
K#sub[A] &== K#sub[B]
$

== Man-in-the-Middle Attack

= Module 4: Message Digest and Hash Functions
== Requirements for Hash Functions
A Hash Function $H$ accepts a variable length block of data $M$ as input and
produces a fixed size result $h = H(M)$ referred to as a *hash value* or *hash code*.

A *Cryptographic Hash Function* for which it is computationally infesible to find:
1. $M$ which maps to a predefined $h$
2. $(M#sub[1], M#sub[2])$ which map to same $h$
== Security of Hash Functions
== Message Digest (MD5)
*Setup*
$
"Input":& M\
"Message Length":& 2^64 "bits" \
"Output":& 128 "bits"\
$

*Steps*:
1. *Padding*
Padding bits $P$:
$
    P = 1 #sym.dot.c sum_(i)^n 0_(i) 
$
Padding is *always* added, even if:
$
    O = 512 #sym.dot.c x = M + 64 "bits" | x #sym.in [1, #sym.infinity) 
$
Examples: ${10, 100, 1000}$

Output:
$
    O_(1) = M + P
$

2. *Append Length*

$
    L = "len"(M) | "expressed in 64 bits"
$

Then,

$
    O_(2) = O_(1) + L
$

Output:
$
    O_(2) =& O_(1) + L\
    O_(2) =& M + P + L\
$

3. *Divide into 512 bit blocks*

$
    O_(3) &= sum_(i)^(n) a_(i) | "where len(a) == 512"\
    O_(3) &= {a_(1), a_(2), . . , a_(n)}\
$

4. *Initialize Chaining Variable*

Chaining variables: ${A, B, C, D}$ are initialised, each 32 bits.
#align(center)[
    #table(
        columns: (auto, auto, auto, auto, auto),
        stroke: 0.5pt,
        [*A*], [01], [23], [45], [67],
        [*B*], [89], [AB], [CD], [EF],
        [*C*], [FE], [BC], [DA], [98],
        [*D*], [76], [54], [32], [10],
    )
]

5. *Process Block*
There are *4* rounds.

Let ${a, b, c, d} = {A, B, C, D}$

Divide 512 bits in sub-blocks of 32 bits each (16 sub-blocks):
$
    a_(i) = sum_(j=1)^(16) b_(j) | "where len(b) == 32 bits"
$

Initialize constant _t_: [u32; 64]

Then round function:
$
    a b c d` =  f_(r)(a b c d, {b_(1), .., b_(16)}, t)
$
== Secure Hash Function (SHA)
#image("./sha-512.png")
== Birthday Attack
== HMAC
