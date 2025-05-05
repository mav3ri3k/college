import java.io.*;
import java.net.*;

class Client {
  public static void main() {
    Socket client_socket = new Socket(8080);

    BufferReader in = new BufferReader(new InputStreamReader(client_socket.getInputStream()));
    PrintWriter out = new PrintWriter(client.getOutputStream, true);

    String res = in.readline();
    out.println("Message");

    client_socket.close();
  }
}
