#include <stdio.h>
#include <stdlib.h>

// Process structure
typedef struct {
  int id;
  int arrival_time;
  int burst_time;
  int priority;
  float waiting_time;
  float turnaround_time;
} Process;

// Queue structure
typedef struct {
  int capacity;
  int size;
  Process *processes;
} Queue;

// Function to create a new queue
Queue *createQueue(int capacity) {
  Queue *queue = (Queue *)malloc(sizeof(Queue));
  queue->capacity = capacity;
  queue->size = 0;
  queue->processes = (Process *)malloc(capacity * sizeof(Process));
  return queue;
}

// Function to add a process to the queue
void enqueue(Queue *queue, Process process) {
  // If the queue is full, double the capacity
  if (queue->size == queue->capacity) {
    queue->capacity *= 2;
    queue->processes =
        (Process *)realloc(queue->processes, queue->capacity * sizeof(Process));
  }

  // Find the correct position to insert the process based on priority and
  // arrival time
  int i;
  for (i = 0; i < queue->size; i++) {
    if (queue->processes[i].priority < process.priority ||
        (queue->processes[i].priority == process.priority &&
         queue->processes[i].arrival_time > process.arrival_time)) {
      break;
    }
  }

  // Shift the processes after the insertion position
  for (int j = queue->size; j > i; j--) {
    queue->processes[j] = queue->processes[j - 1];
  }

  // Insert the process at the correct position
  queue->processes[i] = process;
  queue->size++;
}

// Function to process the queue and calculate the waiting and turnaround times
void processQueue(Queue *queue) {
  float total_waiting_time = 0;
  float total_turnaround_time = 0;

  printf("Process ID\tArrival Time\tBurst Time\tPriority\tWaiting "
         "Time\tTurnaround Time\n");

  float current_time = 0;
  for (int i = 0; i < queue->size; i++) {
    // Wait for the process to arrive if necessary
    if (current_time < queue->processes[i].arrival_time) {
      current_time = queue->processes[i].arrival_time;
    }

    // Calculate the waiting and turnaround times
    queue->processes[i].waiting_time =
        current_time - queue->processes[i].arrival_time;
    queue->processes[i].turnaround_time =
        queue->processes[i].waiting_time + queue->processes[i].burst_time;

    // Update the total waiting and turnaround times
    total_waiting_time += queue->processes[i].waiting_time;
    total_turnaround_time += queue->processes[i].turnaround_time;

    // Print the process information
    printf("%11d\t%12d\t%11d\t%9d\t%13.2f\t%15.2f\n", queue->processes[i].id,
           queue->processes[i].arrival_time, queue->processes[i].burst_time,
           queue->processes[i].priority, queue->processes[i].waiting_time,
           queue->processes[i].turnaround_time);

    // Update the current time
    current_time += queue->processes[i].burst_time;
  }

  printf("\nTotal Waiting Time: %.2f\n", total_waiting_time);
  printf("Average Waiting Time: %.2f\n", total_waiting_time / queue->size);
  printf("Total Turnaround Time: %.2f\n", total_turnaround_time);
  printf("Average Turnaround Time: %.2f\n",
         total_turnaround_time / queue->size);
}

int main() {
  int n_processes;
  printf("Enter the number of processes: ");
  scanf("%d", &n_processes);

  Queue *queue = createQueue(n_processes);

  printf("Enter the process details:\n");
  for (int i = 0; i < n_processes; i++) {
    Process process;
    process.id = i + 1;
    printf("Process %d:\n", process.id);
    printf("Arrival Time: ");
    scanf("%d", &process.arrival_time);
    printf("Burst Time: ");
    scanf("%d", &process.burst_time);
    printf("Priority: ");
    scanf("%d", &process.priority);
    enqueue(queue, process);
  }

  processQueue(queue);

  // Free the memory used by the queue
  free(queue->processes);
  free(queue);

  return 0;
}
