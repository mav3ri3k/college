#import "lib.typ": *


#show: doc => assignment(
  title: "Project Work",
  course: "Information Security",
  // Se apenas um autor colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "12 Novemeber 2024",
  doc,
)


= Team Members
#cblock[
    *Divya Lakshi H*: 22BCE2544\
    *Apurva Mishra*: 22BCE2791\
    *Taware Harsh Vilas*: 22BCE2878\
    *Tanmay Patel*: 22BCE2911\
    *Mohammad Faiz Khan*: 22BCE2995\
]
= Title
P2P Chat

= Aim
To develop a peer to peer chat application which is end to end
encrypted and does not require login to ensure complete user privacy

= Objectives
- *Key Exchange*: Implement a method for securely exchanging encryption keys between peers using the shared secret phrase without relying on a central server.
- *End-to-End Encryption*: Ensure all messages are encrypted from sender to recipient, preventing any intermediary from reading them.
- *Peer Discovery*: Develop a decentralized mechanism for peers to discover each other using the shared secret phrase.
- *Secure Messaging*: Implement a secure messaging protocol that handles message delivery, order, and potential loss.
- *User Interface*: Create a user-friendly web interface for sending and receiving messages, managing connections, and setting up the chat.

= Planned Salient Features
- *Secret Phrase Based Connection*: Users connect using a shared secret phrase.
- *End-to-End Encryption*: All communication is end-to-end encrypted.
- *Decentralized Peer Discovery*: No central server is required for finding peers.
- *Open Source*: Make the application open source to allow for community review and improvement.

= Required Technologies
- *Programming Language*: Rust and Javascript
- *Networking Libraries*: #link("https://www.iroh.computer/")[Iroh]
- *User Interface Framework*: React
- *Version Control*: Git

