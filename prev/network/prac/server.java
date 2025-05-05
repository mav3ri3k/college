import java.io.*
import java.net.*;

public class Server {
  public static void main() {
    ServerSocket server_socket = new ServerSocket(8080);
    System.out.println("Server listening on 8080");

    Socket client_socket = server_socket.accpet();
    System.out.println("Client connection established");

    BufferReader in = new BufferReader(new InputStreamReader(client_socket.getInputStream()));
    PrintWriter out = new PrintWriter(client_socket.getOutputStream(), true);

    String message = in.readline();
    System.out.println("Client says: " + message);

    out.println("Message received");

    client_socket.close();
    server_socket.close();
  }
}
