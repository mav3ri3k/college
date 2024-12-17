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
