#import "lib.typ": *
#import "@preview/fletcher:0.5.4":*

#show: doc => assignment(
  title: "Notes",
  course: "Cryptography and Network Security",
  // Se apenas um autor colocar , no final para indicar que é um array
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
    B &= g^y "mod"n\
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
== Security of Hash Functions
== Message Digest (MD5)
== Secure Hash Function (SHA)
== Birthday Attack
== HMAC
