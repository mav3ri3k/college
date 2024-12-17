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
