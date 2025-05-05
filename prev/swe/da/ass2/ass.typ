#import "lib.typ": *
#import "@preview/pintorita:0.1.3"

#show raw.where(lang: "pintora"): it => pintorita.render(it.text)

#show: doc => assignment(
  title: "Digital Assignment - II",
  course: "Software Engineering Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "17 February, 2025",
  doc,
)

= Entity Relationship Diagram (Structural Modeling) 
```pintora
erDiagram
    title: Social Media Platform

    User {
    uuid id PK
    string username
    string email
    string profile_pic
    date join_date
    }

    User ||--o{ Post : create
    User ||--o{ FOLLOW : follower
    FOLLOW ||--o{ User : following

    Post {
    uuid id PK
    sting content
    datetime timestamp
    sting media_url
    uuid user_id FK
    }

    User ||--o{ Comment : writes
    Post ||--|{ Comment : has
    
    Comment {
    uuid id PK
    string content
    datetime timestamp
    uuid user_id FK
    uuid post_id FK
    }

    User ||--o{ Like : makes
    Post ||--|{ Like : receives

    Like {
    uuid id PK
    datetime timestamp
    uuid user_id FK
    uuid post_id FK
    }
```
= Content Flow Diagram, DFD (Functional Modeling)
#image("q2.png")
#align(center, [Social Media Platform])

= State Transition Diagram (Behavioral Modeling)
```pintora
componentDiagram
    title: Post on Social Media Platform
    @param edgesep 115
    @param useMaxWidth true

    Start --> [Draft]

    [Draft] --> [PendingReview] : Submit Post

    [PendingReview] --> [Published] : Approved
    [PendingReview] --> [Rejected] : Violates Guidelines

    [Published] --> [Trending] : High Engagement
    [Published] --> [Archieved] : Time Threshold
    [Published] --> [Hidden] : Reported

    [Archieved] --> End

    [Trending] --> [Published] : Engement Drops

    [Rejected] --> [Draft] : Edit

    [Hidden] --> [PendingReview] : Under Review

```

