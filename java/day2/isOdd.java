import java.util.Scanner;

public class isOdd{
  public static void main(String args[]) {
    int sum = 0; 
    for (int i = 0; i < 5; i++) { 
      System.out.println("Enter the number: "); 
      Scanner sc = new Scanner(System.in);
      int number = sc.nextInt(); 
      if (number % 2 != 0) {
        sum += number; 
      }
    }
    System.out.println(sum);
  }
}
