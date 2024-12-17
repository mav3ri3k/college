#import "lib.typ": *
#import "dvd.typ": *
#import "@preview/showybox:2.0.1": showybox

#show image: it => block(
  radius: 5pt,
  clip: true,
)[#it]

#let code_block(ctitle: "Here", cbody) = {
  showybox(
    frame: (
      border-color: black,
      title-color: red.lighten(60%),
      body-color: luma(230),
    ),
    title-style: (
      color: black,
      weight: 100pt,
    ),
    body-style: (
      align: left,
    ),
    sep: (
      dash: "dashed",
    ),
    shadow: (
      offset: (x: 1pt, y: 1pt),
      color: black.lighten(70%),
    ),
    breakable: true,
    text(weight: "bold")[#ctitle],
    cbody,
  )
}
#show raw: name => if name.block [
  #name
] else [
  #box(
    fill: luma(230),
    outset: (x: 2pt, y: 3pt),
    radius: 4pt,
  )[#name]
]

#show raw: name => if name.block [
  #block(
    fill: luma(230),
    inset: 4pt,
    radius: 10pt,
  )[#name]
] else [
  #box(
    fill: luma(230),
    outset: (x: 2pt, y: 3pt),
    radius: 10pt,
  )[#name]
]

#show link: lnk => [
  #text(fill: blue)[
    #underline(lnk)
  ]
]

#show: doc => report(
  title: "Digital Assignment - IV",
  course: "Operating System Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  authors: ("Apurva Mishra, 22BCE2791",),
  date: "20 September 2024",
  doc,
)

#let view(
  question,
  output,
  output_size,
  raw_code,
) = {
  problem[
    #question
  ]
  grid(
    inset: 3pt,
    columns: (auto, auto),
    /*
    align(center)[
      #image(output, height: output_size, fit: "stretch")
    ],
    */
    raw_code,
  )
}

= Questions

#problem[
  Write a LINUX C programme to enable the inter process communication mechanism between the process writer and reader by utilising shared memory.

  Note: Create two IPC programmes that use shared memory. Program 1 will create the shared segment, attach it to it, and write some content into it. Then, Program 2 will connect to the shared segment and read the value that Program 1 has written.
]

#code_block(
  ctitle: "server.c (writer)",
  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <sys/shm.h>
  #include <unistd.h>

  #define SHM_SIZE 1024

  int main() {
    int i;
    void *shared_memory;
    char buff[100];
    int shmid;

    shmid = shmget((key_t)2323, SHM_SIZE, 0666 | IPC_CREAT);

    printf("Server: Shared memory key is %d\n", shmid);
    shared_memory = shmat(shmid, NULL, 0);

    printf("Server: Process attached at %p\n", shared_memory);

    while (true) {
      printf("\nServer: Enter data to write to shared memory: \n");

      read(0, buff, 100);
      strcpy(shared_memory, buff);
      printf("Server: You wrote: %s\n", (char *)shared_memory);
    }

    printf("\nServer: Exiting...\n");
    return 0;
  }
  ```,
)

#pagebreak()

#code_block(
  ctitle: "client.c (reader)",
  ```c
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <sys/shm.h>
  #include <unistd.h>

  #define SHM_SIZE 1024

  int main() {
    int i;
    void *shared_memory;
    char buff[100];
    int shmid;

    shmid = shmget((key_t)2323, SHM_SIZE, 0666);
    printf("Client: Shared memory key is %d\n", shmid);

    shared_memory = shmat(shmid, NULL, 0);
    printf("Client: Process attached at %p\n", shared_memory);

    int count = 1;
    while (true) {
      printf("\nClient: %d[+] Data read from shared memory: %s\n", count,
             (char *)shared_memory);
      sleep(15);
      count += 1;
    }

    printf("\nClient: Exiting...\n");
    return 0;
  }
  ```,
)

#text(size: 15pt, weight: "bold")[Output: Sender and Receiver]\
Here `server.c` is writing to the shared memory and `client.c` reads from the shared memory.
Also, `just sr` and `just cr` are build scripts for server and client respectively.
#link("https://andrewkelley.me/post/zig-cc-powerful-drop-in-replacement-gcc-clang.html")[zig cc] is used as the C compiler on MacOS.\
#text()[]\
#pagebreak()

#image("q1s.png")

#image("./q2c.png")

