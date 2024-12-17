#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct item {
  int label;
  char symbol[3];
  int address;
  struct item *next;
};

struct item *createItem(int label, char *symbol, int address) {
  struct item *newItem = (struct item *)malloc(sizeof(struct item));

  newItem->label = label;
  strcpy(newItem->symbol, symbol);
  newItem->address = address;
  newItem->next = NULL;

  return newItem;
}

void insertAtBeginning(struct item **head, int label, char *symbol,
                       int address) {
  struct item *newItem = createItem(label, symbol, address);

  newItem->next = *head;
  *head = newItem;
}

// Function to insert an item at the end of the list
void insertAtEnd(struct item **head, int label, char *symbol, int address) {

  struct item *newItem = createItem(label, symbol, address);
  struct item *temp = *head;

  if (*head == NULL) {
    *head = newItem;
    return;
  }

  while (temp->next != NULL) {
    temp = temp->next;
  }
  temp->next = newItem;
}

void deleteItem(struct item **head, int label) {
  struct item *temp = *head, *prev = NULL;

  if (temp != NULL && temp->label == label) {
    *head = temp->next;
    free(temp);
    return;
  }

  while (temp != NULL && temp->label != label) {
    prev = temp;
    temp = temp->next;
  }

  if (temp == NULL) {
    return; // Item not found
  }

  prev->next = temp->next;
  free(temp);
}

void modifyItem(struct item *head, int label, int newLabel, char *newSymbol,
                int newAddress) {
  struct item *temp = head;

  while (temp != NULL && temp->label != label) {
    temp = temp->next;
  }

  if (temp == NULL) {
    return;
  }

  temp->label = newLabel;
  strcpy(temp->symbol, newSymbol);
  temp->address = newAddress;
}

struct item *searchItem(struct item *head, int label) {
  struct item *temp = head;

  while (temp != NULL && temp->label != label) {
    temp = temp->next;
  }

  return temp;
}

void displayList(struct item *head) {
  struct item *temp = head;

  while (temp != NULL) {
    printf("Label: %d, Symbol: %s, Address: %d\n", temp->label, temp->symbol,
           temp->address);
    temp = temp->next;
  }
}

int main() {
  // singly linked list
  struct item *head = NULL;

  insertAtEnd(&head, 1, "A", 100);
  insertAtBeginning(&head, 2, "B", 200);
  insertAtEnd(&head, 3, "C", 300);

  printf("Original list:\n");
  displayList(head);

  modifyItem(head, 2, 20, "BB", 220);

  deleteItem(&head, 1);

  struct item *foundItem = searchItem(head, 3);
  if (foundItem != NULL) {
    printf("\nItem found: Label: %d, Symbol: %s, Address: %d\n",
           foundItem->label, foundItem->symbol, foundItem->address);
  } else {
    printf("\nItem not found\n");
  }

  printf("\nModified list:\n");
  displayList(head);

  return 0;
}
