#import "lib.typ": *
#import "@preview/fletcher:0.5.4":*

#show: doc => assignment(
  title: "Notes",
  course: "Information Security",
  // Se apenas um author colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "CAT - II",
  doc,
)


= Module:3 Operating Systems Security

== Security in Operating System
=== *Functions of OS*
#image("f_os.png")
1. *Enforce Sharing*: Integrity control such as monitors and transaction processes.
2. *IPC*
3. *Protection of Critical Sections*
4  *Interface to Hardware*
5. *User Authentication*
6. *Separate Memory*: Each user program runs in protected portions of memory

=== *Multi-Programming and Shared Use*\
Realizing that two users could interleave access to
the resources of a single computing system, researchers developed concepts such as
scheduling, sharing, and concurrent use. Multiprogrammed operating systems, also
known as monitors, oversaw each program’s execution. Monitors took an active role,
whereas executives were passive. That is, an executive stayed in the background, waiting
to be called into service by a requesting user.

=== *Multitasking*
1. *Process*: Program run by user
2. *Domain*: Resources allocated to process
3. *Threads*: Process consist of one of more threads
4. *Tasks*: Threads can spawn task which are smmaller executable units of code

OS switches between process allocating and deallocating resources as needed.

=== Layered Design
#image("./layered_os.png")

OS also expose certain *hooks* to which programs such as antivirus can
attach to so that they load before user executables.

=== Self Protection by OS
The
default expectation is one level of hardware-enforced separation (two states).
This situation means that an attacker is only one step away from complete
system compromise through a *get-root* exploit.
\
\
This is partly mitigated by the layerd nature of the OS.

=== Tools to implement Security Functions
1. *Audit Logs*: However this is not possible, on scale due to sheer volume of data.
2. *Virtualization*: presenting a user the appearance of a system with only the resources the user is entitled to use.
3. *Honeypot*: system to lure an attacker into an environment that can be both controlled and monitored
4. *Separation*: Using
    - Space: Physical
    - Time: Physical
    - Access Control: Logical
    - Cryptography
#cblock[#text(fill: red)[IMPORTANT]]
5. *Fence*: 
    There are two ways:
    1. Hardware Fence which is hardcoded
    2. *Fence Register* which contains the address of the end of OS and can be reprogrammed.
6. *Base and Bound Register*:
    1. *Base Register*: Starting address of the program memory
    2. *Bound Register*: Upper address limit for the program. 
7. *Tagging Architecture*: Every word of machine memory has one or more word to determine access rights.
8. *Virtual Memory*:
    1. *Segmentation*: A logical contiguous sequence of memory that is can be mapped to actual physical memory. A table must map logical segment to physical address.\
            *Advantages*:
                - Can be easily moved in physical memory without affecting logiacl mapping.
                - A segment can be moved from main memory.
                - Every reference passes through the table, so it can be checked.
                - Table can also be used to maintain access control.
    2. *Paging*: Program and memory are divided in equal size pieces called pages. Program pages are mapped to pages in physical memory and may not be contiguous. More memory efficient management than segments. 

    #figure(
        image("./seg_pag.png"),
        caption: [Segmentation + Paging Combination]
    )
== Security in the design of OS
Secure Design principles:
1. Least privilege
2. Open Design
3. Permission Based
4. Separation of Privileges

=== Simplified Design,
    Security should be incroporated from the start. Keep the design simple to keep it maintaineable and auditable.
=== Layered Design,
    1. Hardware
    2. Kernel
    3. OS
    4. User
With sublayers. This allows system to *evolve* over time.\
\
*Advantages*:
1. *Layered Trust*:\
    Most sensitive tasks are assigned to lower levels, with increasing level of authorization required to access. 
2. *Encapsulation*
3. *Damage Control/Isolation*
4. *Auditability*: Lower layers can be analyzed more critically.

=== Kernelized design,
Security functions are contained in Security Kernel.\
\
*Advantages*:
    1. *Access Control*
    2. *Easier Tracing*
    3. *Modifiability*
    4. *Compactness*
    5. *Verifiability* 
=== Reference Monitor,
This is like a wall. It separates subjects and objects, enforcing that a subject can access only those objects allowed by security policy.\
\
Reference Monitor *must me:*
1. *Tamperproof*
2. *Unbypassable*
3. *Verifiable*

=== Trusted Systems
One with evidence to substantiate the claim it implements
some function or policy. We can say the code is trused if its *rigorously tested and anlysed*.\
\
*Key Characteristics*:
1. _Functional Correctness_
2. _Integrity_
3. _Limited Privileges_

=== Trusted Systems Functions
*Trusted Computing Base (TCS)*: Everything necessary in computer to enforce security policy. 
#figure(
    image("./tcb.png"),
    caption: [Parts on OS mananged by TCB]
)

*TCB Monitors*:
1. *Process Activation*: This involves complete change of register allocation, file access control, (*domains*) etc which are security sensitive.
2. *Execution Domain Switching*: Process running under one *domain* invokes another domain for services.
3. *Memory Protection*: The *domain* (_memory_) is monitored my TCB.
4. *I/O*: Information flowing in and out of *domain*.

Thus _TCB_ manages *domain*:
- Invocation
- Execution
- Access
- Information flowing in and out of it

== Trusted Operating System Design
1. *Layered Design with Security Kernel*
2. *Secure Startup*: When system starts, all security functions must be working properly and no effect remain from previous session.
    Thus not malicious code can block security enforcement.
3. *Trusted Path*: is an unforgeable connection by which the user can be confident of communicating directly with the operating system, not with any fraudulent intermediate application.
4. *Object Reuse*: All space previously allocated is cleared before it is reassigned.
5. *Audit*: All security related changes can be tracked.
== Rootkit.

= Module:4 Security Countermeasures 7 hours
== Design of Firewalls
== Types
== Personal Firewalls
== Configurations
== Network Address Translation
== Data Loss Prevention
== Intrusion Detection and Prevention Systems:
=== Types of IDSs
=== Intrusion Prevention system
=== Intrusion Response
=== Goals of IDSs
=== Strength and Limitations
= Module:5 Database Security
== Database Security
== Database Security Requirements
== Reliability and Integrity
== Sensitive Data
== Types of Disclosures
== Preventing Disclosures
== Inference
== Multilevel Databases
== Multilevel Security
== Database Attacks
== SQL Injection Attacks.
