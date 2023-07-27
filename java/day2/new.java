public class test {
  
  public int power(int base, int power) {
    int number = base; 
    for (i=0; i<power; i++) {
      number *= base;
    }
    return number
  } 

  public static void new(String arg[]) {
    int out = power(3,2);
    System.out.println(out);
  }

}
