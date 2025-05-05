#import "lib.typ": *


#show: doc => assignment(
  title: "Digital Assignment - I",
  course: "Software Engineering Lab",
  // Se apenas um author colocar , no final para indicar que é um array
  author: "Apurva Mishra: 22BCE2791",
  date: "14 April, 2025",
  doc,
)

= Question

#cblock[
  Analysis and Identification of the suitable process models 
\
]


== Answer
1. Analysis:
  - Goal: Create a platform for users to connect, share content (text, images, short videos), and interact.
  - Target Audience: Young adults (18-30).
  - Core Features (Initial MVP): User registration/login, profile creation, posting text/images, a chronological feed of followed users, following/follower system, basic notifications (new follower, like).
  - Key Challenges: Scalability, content moderation, user engagement, data privacy, competition.
  - Environment: Dynamic market, user expectations evolve rapidly, technology changes quickly. Need for flexibility and rapid feedback incorporation.
2. Identification of Suitable Process Models:
  - Waterfall: Not suitable. Requirements are likely to change based on user feedback and market trends. A rigid, sequential process would be too slow and inflexible.
  - Agile (Scrum/Kanban): Highly suitable.
    1. Iterative Development: Allows building the platform in small increments (sprints in Scrum), releasing functional pieces early and often (MVP first, then add features like video, groups, messaging).
    2. Flexibility: Easily adapts to changing requirements and user feedback.
    3. Collaboration: Emphasizes close collaboration between developers, designers, product owners, and stakeholders.
    4. Feedback Loops: Regular reviews and retrospectives help incorporate feedback quickly.
  - Spiral: Could be considered if initial risk analysis (e.g., major privacy regulations, core scaling technology) is paramount, but Agile generally offers better speed for feature delivery in this domain.
*Conclusion:* An Agile methodology, likely Scrum, is the most suitable process model for developing a social media platform due to the need for flexibility, rapid iteration, and user feedback incorporation.

= Question

#cblock[
  Work Break-down Structure (Process Based, Product Based, Geographic 	Based 	and 	Role 	Based) 	and  Estimations 
\
]
== Answer
#figure(
image("./wbs.png"),
caption: [Work Break-down Structure]
)

= Question

#cblock[
  Story Boarding
\
]

== Frame 1-4
*Frame 1:* Main Feed\
- Scene: User is logged in and on the main feed.\
- UI Elements: Feed with user posts, navigation bar (Home, Profile, etc.), and a prominent “+” or "Create Post" button.
- Action: User notices the button and is about to tap it.

*Frame 2:* Tapping Create Post\
- Scene: Close-up of the user's finger tapping the "Create Post" button.
- UI Elements: Button animates with feedback (e.g., a ripple or highlight effect).
- Action: Button press triggers a transition to post type selection.

*Frame 3:* Choose Post Type\
- Scene: A modal or full screen shows post type options.
- UI Elements: Buttons labeled "Text Post", "Image Post", and "Video Post".
- Action: User selects "Image Post" by tapping it.

*Frame 4:* Photo Selection
- Scene: The device’s photo library (or camera interface) is now open.
- UI Elements: Image grid or camera view, “Cancel” and “Select” buttons.
- Action: User taps on a photo from the gallery.

#figure(
image("./14.png", height: 40%),
caption: [Frames 1-4]
)

== Frame 5-8
*Frame 5:* Image Preview and Caption UI\
- Scene: Image preview screen.
- UI Elements: The selected image is centered. Below it:
  1. A caption input field.
  2. Optional filter selection or image editing tools.
- Action: User begins typing a caption.

*Frame 6:* Caption Added\
- Scene: Caption field filled in.
- UI Elements: Caption reads “Beautiful sunset!”.
- Action: User taps the “Post” button.

*Frame 7:* Posting the Content\
- Scene: Posting action begins.
- UI Elements: A loading spinner or progress bar appears.
- Action: Upload in progress animation.

*Frame 8:* Post Successful\
- Scene: User is back on the feed.
- UI Elements: New post is at the top of the feed, with:
  1. Selected image.
  2. Caption below.
  3. Small “Post successful!” toast message appears briefly.
- Action: User sees their content live.

#figure(
image("./58.png", height: 40%),
caption: [Frames 5-8]
)


== Frame 9
*Frame 9:* End State
- Scene: User scrolls through the feed with satisfaction.
- UI Elements: Engagement icons (like, comment, share), and smooth animation returning to normal feed flow.
- Action: Flow complete.

#figure(
image("./9.png", height: 30%),
caption: [Frame 9]
)
