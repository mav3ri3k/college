import java.util.Scanner;

public class Main {
  public static void main() {

  }

  public studentAllowed() {
    Scanner sr = new Scanner(System.in);
    Double totalClass = sr.nextDouble();
    Double attendedClass = sr.nextDouble();
    boolean medical = sr.nextBoolean();
    sr.close();

    Double percentAttendanceDouble = attendedClass/totalClass;
    
    if (percentAttendanceDouble >= .75 && medical) {
      System.out.println("Allowed");
    } else {
      System.out.println("Not Allowed");
    }

  }

  public totalOil() {
    Scanner sr = new Scanner(Systen.in);
    Integer n = sr.nextInt();

    Integer totalWhole = 0;
    Integer totalPart = 0;
    for (i = 0; i < n; i++) {
      totalWhole += sr.nextInt();
      totalPart += sr.nextInt();
  
      while (totalPart >= 1000) {
        totalPart -= 1000;
        totalWhole++;
      }
    }
    System.out.print("Total: " + totalWhole + "L");
    System.out.print(" " + totalPart + "ml\n");
  }

  public height() {
    Scanner scanner = new Scanner(System.in);

    System.out.println("Length : ");
    int length = scanner.nextInt();

    System.out.println("Angle : ");

    double angleInRadians = Math.PI / 180.0 * angle;
    double height = length * Math.sin(angleInRadians);

    // Round the height to two decimal places.
    height = Math.round(height * 100.0) / 100.0;

    if (height > 0) {
      System.out.println("Height: " + height + " feet.");
    } else {
      System.out.println("Too short");
    }
  }

  public factor() {
    Scanner scanner = new Scanner(System.in);

    System.out.println("Enter a number: ");
    int n = scanner.nextInt();

    List<Integer> factors = new ArrayList<>();

    for (int i = n; i > 0; i /= 10) {
      int digit = i % 10;

      if (n % digit == 0) {
        factors.add(digit);
      }
    }

    if (factors.isEmpty()) {
      System.out.println("No factors");
    } else {
      for (int i = factors.size() - 1; i >= 0; i--) {
        System.out.println(factors.get(i));
      }
    }
  }

  public bookPrice() {
    Scanner scanner = new Scanner(System.in);

    System.out.println("Enter the product code: ");
    int productCode = scanner.nextInt();

    System.out.println("Enter the order amount: ");
    int orderAmount = scanner.nextInt();

    int discount = 0;
    if (productCode == 1 && orderAmount > 1000) {
      discount = 10;
    } else if (productCode == 2 && orderAmount > 750) {
      discount = 5;
    } else if (productCode == 3 && orderAmount > 500) {
      discount = 10;
    }

    int netAmount = orderAmount - (orderAmount * discount / 100);

    System.out.println("Total Amount: " + netAmount);
  }

  public tollCharge() {
    Scanner scanner = new Scanner(System.in);

    System.out.println("Enter the kilometers travelled: ");
    int kilometersTravelled = scanner.nextInt();

    int tollCharge = 0;
    if (kilometersTravelled < 1000) {
      tollCharge = 0;
    } else if (kilometersTravelled < 2000) {
      tollCharge = 50;
    } else if (kilometersTravelled < 4000) {
      tollCharge = 250;
    } else if (kilometersTravelled < 6000) {
      tollCharge = 350;
    } else {
      tollCharge = 500;
    }

    System.out.println("Toll Charge: " + tollCharge);
  }
}
