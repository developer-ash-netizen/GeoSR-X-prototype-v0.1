import sys
from backend.server import serve
if __name__ == "__main__":
    host = "127.0.0.1"; port = 8000
    for a in sys.argv[1:]:
        if a.isdigit(): port = int(a)
        elif a.startswith("--host="): host = a.split("=", 1)[1]
    serve(host, port)
