= Module 1
*Marks: 10*
  - Discuss Models
  - Hierarchy Aspect
  - Relevant Technologies

== Limitations of CIoT (Cloud to Edge): BLURS
- Bandwidth
- Latency
- Uninterpted Connectivity
- Resource Constraint (Power)
- Security

== Advantages of Fog
- Hierarchial
- Flexible
- Scalable

== Cloudlet Computing
Smaller Compute Resources compared to data centres deployed closer to edge
resources aimed for mobile applications with lower latency.
(Cluster of small data centers generally one hop away)

== Virtualisation ??

== Mist Computing ??

== Advantages of Fog and Edge Computing: SCALE
- Security
- Cognition
- Agility
- Latency
- Efficiency

== What FEC Provide
- Storage: Cache
- Compute: VM
- Acceleration: FPGA/GPU
- Networking: TCP/UDP (Vertical Networking), Bluetooth/Zigbee (Horizontal Networking)
- Control: Deployment, Actuation

== Hierarchy
- Core Network
- Inner Edge: LAN
- Middle Edge: Fog
- Outer Edge: Sensors and Actuators

== Buinses Models
- X as service (XaaS): Provide hardware, infrastrucre, software as service
- Application: Efficient Data Processing, or solutions to a problem
- Support

== Challenges
- Vendor Lockin
- Vertical Integration across all the layers
- Security Risks
- Increased Complexity
- Interoperability across hardware acrsos the layers
- Limited Support

= Module 2
*10 Marks*
- C2F2T: Uses Casess & Metrics
- Formulas


== Federated Edge Resources
Process of sharing and connecting resources at the edge of network.


=== Challenges
1. Networking
  - User Mobility
  - QoS
  - Achieving service centric model
  - Ensuring reliability
  - Managing Multiple administrative domains

  Software Defined Network *(SDN)* is potential solution for orchestrating edge resources.

2. Management
  - Discovery of edge nodes
  - Deployment of service and application
  - Migration service
  - Load Balancing

  Rapid service migration between nodes cause high overhead and unsuitability.

== Analytical Models

=== Other Models
- *Fuzzy Ontology:* Long term fault classification and improving diagnosis accuracy
- *Bayesian Probability:* Possible futures states with level of uncertainty
- *Markov Chain:* Energy consumption
- *Linear Programming*: Cost in IoT, scheduling and task allocation.
- *Petri Nets*: Data flow in the system and verifying task correctness.

= Module 3
*10 Marks*
- 5/7 Layers Architecture
- Network Slicing (Scenario/Application Based)

== QoS Requirements
- *Low Latency Communication*
- *High Bandwidth*
- *High Energy Efficiency*

== Network Slicing
Partition physical network in multiple isolated logical networks

Networkd as Service (*NaaS*)

*Benefits*:
- Cost Efficiency
- Flexibility
- Scalability


*Layers*
#cblock[Infrastructure Layers]
#cblock[
1. Software Defined Network
2. Network Function Virtualisation
3. Virtualisation
]
#cblock[Service and Application]

#cblock[Over all this: Management]

== IoV
#cblock[
  #cblock[
  *Vehicle to*
    1. Vehicle: IEEE Wave
    2. Roadside: Ethernet
    3. Person: CarPlay/Bluetooth
    4. Internet/Cellular Tower: Wifi 4G/LTE
    5. Everything
  ]
  #cblock[
    Sensor and Actuator
    Roadside to Roadside
    Roadside to Personal Device
  ]
]

=== 5 Layers
#image("5l.png")

=== 7 Layers
#image("./7l.png")


=== Protocols
#image("./operation.png")
#image("./proto.png")

#cblock[
  #cblock[
    *Business Layer*
    WAVE
  ]
  #cblock[
    *App Layer*
    Resource Handler
  ]
  #cblock[
    *AI Layer*
    - Big Data Analysis
    - CALM: Communication Access of Land Module
    - HSM: Hardware Security Module

  ]
  #cblock[
    *Coordination Layer*
    - TCP/UDP
    - WASP
    - MAC
  ]
  #cblock[
    *Perception*
    - Physical
    - Ethernet
    - GSM
    - 4G/LTE
  ]
]

=== Challenges
- Location Accuracy
- Location Privacy
- Location Verification
- Operational Management
