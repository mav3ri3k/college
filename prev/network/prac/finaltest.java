import java.io.*;
import java.net.*;

class Server {
  public static void main() {
    ServerSocket server_socket = new ServerSocket(8080);
    System.out.println("Server connected");

    Socket client_socket = server_socket.accept();

    BufferedReader in = new BufferReader(new InputStreamReader(client_socket.getInputStream()));
    PrintWriter out = new PrintWriter(client_socket.getOutputStream(), true);

    String rep = BufferReader.readline()
    out.println("Message");

    client_socket.close();
    server_socket.close();
    }
}
