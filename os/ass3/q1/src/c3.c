#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int uuid;
  int time; // burst time
} Job;

Job job_new(int uuid, int time) {
  Job job;
  job.uuid = uuid;
  job.time = time;
  return job;
}

typedef struct {
  int capacity;
  int length;
  Job *jobs;
} Queue;

Queue *queue_new() {
  Queue *queue = malloc(sizeof(Queue));
  Job *jobs = malloc(sizeof(Job) * 10);
  queue->capacity = 5;
  queue->length = 0;
  queue->jobs = jobs;
  return queue;
}

bool queue_is_empty(Queue *queue) {
  if (queue->length <= 0) {
    return true;
  }
  return false;
}

void increase_capacity(Queue *queue) {
  Job *new_jobs = malloc(sizeof(Job) * (queue->capacity + 5));
  for (int i = 0; i < queue->length; i++) {
    new_jobs[i] = queue->jobs[i];
  }
  queue->capacity += 5;
  free(queue->jobs);
  queue->jobs = new_jobs;
}

void queue_add_job(Queue *queue, Job job) {
  if (queue->length == queue->capacity) {
    increase_capacity(queue);
  }
  for (int i = 0; i < queue->length; i++) {
    if (queue->jobs[i].time > job.time) {
      for (int j = queue->length; j >= i; j--) {
        queue->jobs[j] = queue->jobs[j - 1];
      }
      queue->jobs[0] = job;
      queue->length += 1;
      return;
    }
  }
  queue->jobs[queue->length] = job;
  queue->length += 1;
}

void sort_queue_by_burst_time(Queue *queue) {
  for (int i = 0; i < queue->length - 1; i++) {
    for (int j = 0; j < queue->length - i - 1; j++) {
      if (queue->jobs[j].time > queue->jobs[j + 1].time) {
        // Swap jobs
        Job temp = queue->jobs[j];
        queue->jobs[j] = queue->jobs[j + 1];
        queue->jobs[j + 1] = temp;
      }
    }
  }
}

void input_jobs(Queue *queue) {
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
  sort_queue_by_burst_time(queue);
}

int queue_process(Queue *queue) {
  float total = 0;
  float twaiting = 0;
  printf("Process Burst_Time Waiting_Time Turnaround_Time\n");
  fflush(stdout);
  for (int i = 0; i < queue->length; i++) {
    printf("%6d %10d %12.2f %15.2f\n ", queue->jobs[i].uuid,
           queue->jobs[i].time, total, queue->jobs[i].time + total);
    twaiting += total;
    total += queue->jobs[i].time;
  }
  printf("\nTotal Waiting Time: %.2f\n", twaiting);
  printf("Average waiting time: %.2f\n", twaiting / queue->length);
  return total;
}

int main() {
  Queue *queue = queue_new();
  input_jobs(queue);
  queue_process(queue);
  free(queue->jobs);
  free(queue);
  return 0;
}
