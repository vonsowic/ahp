import server
import socket

if __name__ == "__main__":
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()  # Get local machine name
    port = 50001
    client.connect((host, port))

    with open("test.xml", 'rb') as file:
        text = file.read()
        client.send(text)

    response = client.recv(4096)
    print(response)
    client.close()
