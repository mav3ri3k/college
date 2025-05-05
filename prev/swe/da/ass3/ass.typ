#import "lib.typ": *
#import "@preview/pintorita:0.1.3"
#import "@preview/fletcher:0.5.6": *

#show raw.where(lang: "pintora"): it => pintorita.render(it.text)

#show: doc => assignment(
  title: "Digital Assignment - III",
  course: "Software Engineering Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "10 March, 2025",
  doc,
)

= Use Case Model


#diagram(
    node-stroke: 1pt, // outline node shapes
    edge-stroke: 1pt, // make lines thicker
    node-corner-radius: 15pt,
    node((-4,7), [User], name: <u>, corner-radius: 0pt),
    node((2,2), [Admin], name: <a>, corner-radius: 0pt),

    node((1, 1), [*Platform Management*], name: <pm>, stroke: 0pt),
    node((1, 1.5), [Moderate Content], name: <mc>),
    node((1, 2), [Ban/Unban User], name: <buu>),
    node(enclose: (<pm>, <mc>, <buu>), stroke: teal, inset: 10pt, snap: false),

    
    node((1, 3), [*Feed & Posts*], name: <fp>, stroke: 0pt),
    node((1, 3.5), [Delete Post], name: <dp>),
    node((1, 4), [Report Post], name: <rp>),
    node((1, 4.5), [Comment], name: <c>),
    node((1, 5), [Like], name: <l>),
    node((1, 5.5), [Share Post], name: <sp>),
    node((1, 6), [Create Post], name: <cp>),
    node(enclose: (<fp>, <dp>, <rp>, <c>, <l>, <sp>, <cp>), stroke: teal, inset: 10pt, snap: false),

    node((1, 7), [*Connections*], name: <conn>, stroke: 0pt),
    node((1, 7.5), [Block User], name: <blk>),
    node((1, 8), [Follow/Unfollow], name: <fol>),
    node(enclose: (<conn>, <blk>, <fol>), stroke: teal, inset: 10pt, snap: false),

    node((1, 9), [*Profile & Groups*], name: <pro>, stroke: 0pt),
    node((1, 9.5), [Create/Join Group], name: <grp>),
    node((1, 10), [Edit Profile], name: <e_prof>),
    node(enclose: (<pro>, <grp>, <e_prof>), stroke: teal, inset: 10pt, snap: false),

    node((1, 11), [*Messaging*], name: <msg>, stroke: 0pt),
    node((1, 11.5), [Send Message], name: <s_msg>),
    node(enclose: (<msg>, <s_msg>), stroke: teal, inset: 10pt, snap: false),

    node((1, 13), [*Social Media Platform*], stroke: 0pt),
    edge(<a>, <mc>),
    edge(<a>, <buu>),

    edge(<a>, <dp>, bend: +35deg),

    edge(<u>, <rp>),
    edge(<u>, <c>),
    edge(<u>, <l>),
    edge(<u>, <sp>),
    edge(<u>, <cp>),

    edge(<u>, <blk>),
    edge(<u>, <fol>, bend: -10deg),

    edge(<u>, <grp>, bend: -25deg),
    edge(<u>, <e_prof>, bend: -30deg),

    edge(<u>, <s_msg>, bend: -40deg),
)


= Class Model Diagram
\
\
_Diagram on Next Page_

```pintora
classDiagram
    title: Social Media Platform
    class User {
        -int userID
        -string username
        -string passwordHash
        -string email
        -string profilePictureURL
        -string bio
        -datetime registrationDate
        +createUser()
        +updateUser()
        +deleteUser()
        +follow(User user)
        +unfollow(User user)
        +block(User user)
        +sendMessage(User user, Message message)
        +getFollowers() List~User~
        +getFollowing() List~User~
        +getBlockedUsers() List~User~
    }
    class Admin {
        -int adminID
        +moderateContent(Post post)
        +banUser(User user)
        +unbanUser(User user)
    }
  
    class Post {
        -int postID
        -string content
        -datetime timestamp
        -string mediaURL
        -PostType postType
        +createPost()
        +updatePost()
        +deletePost()
        +like(User user)
        +unlike(User user)
        +comment(User user, Comment comment)
        +getLikes() List~User~
        +getComments() List~Comment~
    }

    class Comment {
        -int commentID
        -string content
        -datetime timestamp
        +createComment()
        +updateComment()
        +deleteComment()
    }

    class Message {
        -int messageID
        -string content
        -datetime timestamp
        +createMessage()
        +deleteMessage()
    }

    class Group {
        -int groupID
        -string groupName
        -string description
        -datetime creationDate
        +createGroup()
        +updateGroup()
        +deleteGroup()
        +addMember(User user)
        +removeMember(User user)
        +getMembers() List~User~
    }

    class Report {
      -int reportID
      -ReportType reportType
      -string reportText
      -datetime reportDate
      +createReport()
    }

    User --* Post : creates
    User --* Message : sends
    User --* Message : receives
    User -- User : follows
    User -- User : blocks
    User -- Comment : creates
    Post -- Comment : has
    Post -- User : liked by
    Admin -- "0..*" User : manages
    Admin -- "0..*" Post : moderates
    User "0..*" -- "*" Group : joins
    Group "1" -- "0..1" Admin : created by
    User "1" -- "1" Report : reports
    Post "1" -- "0..1" Report : reported_for
    Comment "1" -- "0..1" Report: reported_for


```

= Interaction Model / Sequence Diagram

```pintora
sequenceDiagram
    title: Social Media Platform
    activate User1
    User1 -> Social_Media_Platform: createPost(content, mediaURL=null)
    activate Social_Media_Platform
    Social_Media_Platform -> Post: new(content, timestamp, User1)
    activate Post
    Post --> Social_Media_Platform: postID
    deactivate Post
    Social_Media_Platform --> User1: success(postID)
    deactivate Social_Media_Platform
    deactivate User1

    activate User2
    User2 -> Social_Media_Platform: viewFeed()
    activate Social_Media_Platform
    Social_Media_Platform --> User2: displayPosts([postID, ...])
    deactivate Social_Media_Platform
    User2 -> Social_Media_Platform: likePost(postID)
    activate Social_Media_Platform
    Social_Media_Platform -> Post: like(User2)
    activate Post
    Post --> Social_Media_Platform: updateLikes()
    deactivate Post
    Social_Media_Platform --> User2: success()
    deactivate Social_Media_Platform
    deactivate User2

    activate User3
    User3 -> Social_Media_Platform: viewFeed()
    activate Social_Media_Platform
    Social_Media_Platform --> User3: displayPosts([postID, ...])
    deactivate Social_Media_Platform
    User3 -> Social_Media_Platform: commentOnPost(postID, "Great post!")
    activate Social_Media_Platform
    Social_Media_Platform -> Post: comment(User3, "Great post!")
    activate Post
    Post --> Social_Media_Platform: updateComments()
    deactivate Post
    Social_Media_Platform --> User3: success()
    deactivate Social_Media_Platform
    deactivate User3
```
