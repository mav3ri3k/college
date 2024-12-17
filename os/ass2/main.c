#include <stdio.h>
#include <stdlib.h>

struct job {
  int uuid;
  int time; // burst time
};

struct job job_new(int uuid, int time) {
  struct job job;
  job.uuid = uuid;
  job.time = time;

  return job;
}

struct queue {
  int capacity;
  int length;
  struct job *jobs;
};

struct queue *queue_new() {
  struct queue *queue = malloc(sizeof(struct queue));
  struct job *jobs = malloc(sizeof(struct job) * 10);
  queue->capacity = 5;
  queue->length = 0;
  queue->jobs = jobs;

  return queue;
}

bool queue_is_empty(struct queue *queue) {
  if (queue->length <= 0) {
    return true;
  }
  return false;
}

void increase_capacity(struct queue *queue) {
  struct job *new_jobs = malloc(sizeof(struct job) * (queue->capacity + 5));
  for (int i = 0; i < queue->length; i++) {
    new_jobs[i] = queue->jobs[i];
  }

  queue->capacity += 5;

  free(queue->jobs);
  queue->jobs = new_jobs;
}

void queue_add_job(struct queue *queue, struct job job) {
  if (queue->length == queue->capacity) {
    increase_capacity(queue);
  }
  queue->jobs[queue->length] = job;
  queue->length += 1;
}

void input_jobs(struct queue *queue) {
  int n_job;
  printf("Enter total number of processes:\n");
  scanf("%d", &n_job);

  printf("Enter Process Burst Time:\n");
  for (int i = 0; i < n_job; i++) {
    int burst_time;
    printf("P[%d]:", i + 1);
    scanf("%d", &burst_time);
    queue_add_job(queue, job_new(i + 1, burst_time));
  }
}

int queue_process(struct queue *queue) {
  int total = 0;
  printf("Process Burst_Time Waiting_Time Turnaround_Time\n");
  fflush(stdout);
  for (int i = 0; i < queue->length; i++) {
    printf("%6d %10d %12d %15d\n ", queue->jobs[i].uuid, queue->jobs[i].time,
           total, queue->jobs[i].time + total);

    total += queue->jobs[i].time;
  }

  printf("\nTotal Time: %d\n", total);
  printf("Average waiting time: %d\n", total / queue->length);

  return total;
}

int main() {
  struct queue *queue = queue_new();

  input_jobs(queue);

  queue_process(queue);

  free(queue->jobs);
  free(queue);
  return 0;
}
