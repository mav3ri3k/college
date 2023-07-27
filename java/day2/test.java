public class test {
  
  public static int power(int base, int exp) {
    int number = base; 
    for (int i=1; i<exp; i++) {
      number *= base;
    }
    return number;
  } 

  public static void main (String arg[]) {
    int out = power(2,3);
    System.out.println(out);
  }

}
