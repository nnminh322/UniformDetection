import socket, base64
host="113.163.156.210"; port=555
user="mobiphone"; pw="Hatien2025"
url=f"rtsp://{host}:{port}/4/1"
auth=base64.b64encode(f"{user}:{pw}".encode()).decode()

req = (
f"OPTIONS {url} RTSP/1.0\r\n"
f"CSeq: 1\r\n"
f"Authorization: Basic {auth}\r\n"
f"User-Agent: probe\r\n\r\n"
).encode()

s=socket.socket(); s.settimeout(5)
s.connect((host,port))
s.sendall(req)
try:
    data=s.recv(4096)
    print("RECV:", data[:200])
except Exception as e:
    print("RECV timeout/error:", e)
s.close()
