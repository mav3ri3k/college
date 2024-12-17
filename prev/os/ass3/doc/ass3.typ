#import "lib.typ": report
#import "@preview/dvdtyp:1.0.0": *

#show image: it => block(
  radius: 10pt,
  clip: true,
)[#it]

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

#show: doc => report(
  title: "Digital Assignment - III",
  subtitle: "Operating System Lab",
  // Se apenas um autor colocar , no final para indicar que é um array
  authors: ("Apurva Mishra, 22BCE2791",),
  date: "9 September 2024",
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
  Write a LINUX C Program for the Implementation of shortest remaining time first (SRTF) Scheduling Algorithm.
]
```c
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Job structure
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

// Queue structure
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
      for (int j = queue->length; j > i; j--) {
        queue->jobs[j] = queue->jobs[j - 1];
      }
      queue->jobs[i] = job;
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
    printf("%6d %10d %12.2f %15.2f\n", queue->jobs[i].uuid, queue->jobs[i].time,
           total, total + queue->jobs[i].time);
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
```

#text(size: 15pt, weight: "bold")[Output]
#image("q1.png")


#problem[
  Create a LINUX C program to implement Priority CPU Scheduling with varying arrival times. Processes will be scheduled based on their arrival time and priority.
]

#pagebreak()
#text(weight: "bold")[CPU Scheduling based on Arrival Time]

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int id;
  int arrival_time;
  int burst_time;
  int waiting_time;
  int turnaround_time;
} Process;

void swap(Process *a, Process *b) {
  Process temp = *a;
  *a = *b;
  *b = temp;
}

void sortProcesses(Process *processes, int n) {
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - i - 1; j++) {
      if (processes[j].arrival_time > processes[j + 1].arrival_time) {
        swap(&processes[j], &processes[j + 1]);
      }
    }
  }
}

void processQueue(Process *processes, int n) {
  int current_time = 0;

  for (int i = 0; i < n; i++) {
    if (current_time < processes[i].arrival_time) {
      current_time = processes[i].arrival_time;
    }

    processes[i].waiting_time = current_time - processes[i].arrival_time;
    processes[i].turnaround_time =
        processes[i].waiting_time + processes[i].burst_time;

    current_time += processes[i].burst_time;
  }
}

int main() {
  int n_processes;
  printf("Enter the number of processes: ");
  scanf("%d", &n_processes);

  Process *processes = (Process *)malloc(n_processes * sizeof(Process));

  printf("Enter the process details:\n");
  for (int i = 0; i < n_processes; i++) {
    processes[i].id = i + 1;
    printf("Process %d:\n", processes[i].id);
    printf("Arrival Time: ");
    scanf("%d", &processes[i].arrival_time);
    printf("Burst Time: ");
    scanf("%d", &processes[i].burst_time);
  }

  sortProcesses(processes, n_processes);

  processQueue(processes, n_processes);

  printf(
      "Process ID\tArrival Time\tBurst Time\tWaiting Time\tTurnaround Time\n");
  int total_waiting_time = 0, total_turnaround_time = 0;
  for (int i = 0; i < n_processes; i++) {
    printf("%11d\t%12d\t%11d\t%13d\t%15d\n", processes[i].id,
           processes[i].arrival_time, processes[i].burst_time,
           processes[i].waiting_time, processes[i].turnaround_time);
    total_waiting_time += processes[i].waiting_time;
    total_turnaround_time += processes[i].turnaround_time;
  }

  printf("\nTotal Waiting Time: %d\n", total_waiting_time);
  printf("Average Waiting Time: %.2f\n",
         (float)total_waiting_time / n_processes);
  printf("Total Turnaround Time: %d\n", total_turnaround_time);
  printf("Average Turnaround Time: %.2f\n",
         (float)total_turnaround_time / n_processes);

  free(processes);
  return 0;
}
```
#text(size: 15pt, weight: "bold")[Output]
#image("./q2a.png")
)

#problem[
  Create a LINUX C program to implement Priority CPU Scheduling with varying arrival times. Processes will be scheduled based on their arrival time and priority.
]

#pagebreak()

#text(weight: "bold")[CPU Scheduling based on Arrival Time]
```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int id;
  int arrival_time;
  int burst_time;
  int priority;
  int waiting_time;
  int turnaround_time;
} Process;

void swap(Process *a, Process *b) {
  Process temp = *a;
  *a = *b;
  *b = temp;
}

void sortProcesses(Process *processes, int n) {
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - i - 1; j++) {
      if (processes[j].priority < processes[j + 1].priority ||
          (processes[j].priority == processes[j + 1].priority &&
           processes[j].id > processes[j + 1].id)) {
        swap(&processes[j], &processes[j + 1]);
      }
    }
  }
}

void processQueue(Process *processes, int n) {
  int current_time = 0;
  for (int i = 0; i < n; i++) {
    if (current_time < processes[i].arrival_time) {
      current_time = processes[i].arrival_time;
    }

    processes[i].waiting_time = current_time - processes[i].arrival_time;
    processes[i].turnaround_time =
        processes[i].waiting_time + processes[i].burst_time;

    current_time += processes[i].burst_time;
  }
}

int main() {
  int n_processes;
  printf("Enter the number of processes: ");
  scanf("%d", &n_processes);

  Process *processes = (Process *)malloc(n_processes * sizeof(Process));

  printf("Enter the process details:\n");
  for (int i = 0; i < n_processes; i++) {
    processes[i].id = i + 1;
    printf("Process %d:\n", processes[i].id);
    printf("Arrival Time: ");
    scanf("%d", &processes[i].arrival_time);
    printf("Burst Time: ");
    scanf("%d", &processes[i].burst_time);
    printf("Priority: ");
    scanf("%d", &processes[i].priority);
  }

  sortProcesses(processes, n_processes);

  processQueue(processes, n_processes);

  printf("Process ID\tArrival Time\tBurst Time\tPriority\tWaiting "
         "Time\tTurnaround Time\n");
  int total_waiting_time = 0, total_turnaround_time = 0;
  for (int i = 0; i < n_processes; i++) {
    printf("%11d\t%12d\t%11d\t%9d\t%13d\t%15d\n", processes[i].id,
           processes[i].arrival_time, processes[i].burst_time,
           processes[i].priority, processes[i].waiting_time,
           processes[i].turnaround_time);
    total_waiting_time += processes[i].waiting_time;
    total_turnaround_time += processes[i].turnaround_time;
  }

  printf("\nTotal Waiting Time: %d\n", total_waiting_time);
  printf("Average Waiting Time: %.2f\n",
         (float)total_waiting_time / n_processes);
  printf("Total Turnaround Time: %d\n", total_turnaround_time);
  printf("Average Turnaround Time: %.2f\n",
         (float)total_turnaround_time / n_processes);

  free(processes);
  return 0;
}
```
#text(size: 15pt, weight: "bold")[Output]
#image("./q2b.png")
