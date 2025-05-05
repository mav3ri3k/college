import java.io.*;
import java.net.*;

class Client {
  public static void main() {
    Socket socket = new Socket(8080);
    System.out.println("Connection extanblished on 8080");

    PrintWriter out = new PrintWriter(socket.getOutputStream(), true);

    BufferReader in = new BufferReader(new InputStreamReader(socket.getInputStream()));

    out.println("Message");

    String response = in.readline();
    System.out.println("Server.respoinse");

    socket.close();
  }
}
