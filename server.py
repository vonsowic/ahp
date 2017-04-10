import socket
import main as ahp


def init_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()  # Get local machine name
    port = 50001
    server.bind((host, port))
    server.listen(1)
    while True:
        conn, address = server.accept()  # Establish connection with client.
        print('Got connection from', address)

        while 1:
            req = bytearray(conn.recv(4096)).decode('utf8')
            if not req:
                break

            # print("Request:", req)
            name = ahp.get_response(req)
            print("Name:", name)
            conn.send(bytes(name, encoding='utf8'))
            conn.close()


if __name__ == "__main__":
    init_server()
