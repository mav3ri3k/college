import java.io.*;
import java.net.*;

class Server {
  public static void main() {
    ServerSocket server_socket = new ServerSocket(8080);

    Socket client_socket = server_socket.accept();

    BufferReader in = new BufferReader(new InputStreamReader(client_socket.getInputStream()));
    PrintWriter out = new PrintWriter(client.getOutputStream(), true);

    String response = in.readline();
    out.println("Message received");

    client_socket.close();
    server_socket.close();
  }
}
