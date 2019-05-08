# This class contains methods to handle our requests to different URIs in the app
# import http.server.Base
from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
import time, json
import train

# train

class MyHandler(SimpleHTTPRequestHandler):

    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header("Access-Control-Allow-Origin", "localhost")
        self.end_headers()

    # Check the URI of the request to serve the proper content.
    def do_GET(self):

        if "getVAD" in self.path:
            from urllib.parse import urlparse, parse_qs
            query_components = parse_qs(urlparse(self.path).query)
            if(query_components):
                if (query_components['words'][0]):
                    v, a, d = train.predict(query_components['words'][0])
                self.respond('{"v": ' + str(v) + ', "a": ' + str(a) + ', "d": ' + str(d) + '}')

        else:
            super(MyHandler, self).do_GET()  # serves the static src file by default


    def handle_http(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        return bytes(data, 'UTF-8')


    def respond(self, data):
        response = self.handle_http(data)

        self.wfile.write(response)


if __name__ == '__main__':
    HOST_NAME = "localhost"
    PORT_NUMBER = 8090

    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))