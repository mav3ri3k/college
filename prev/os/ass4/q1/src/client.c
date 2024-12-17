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
