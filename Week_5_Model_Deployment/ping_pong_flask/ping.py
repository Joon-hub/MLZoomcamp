from flask import Flask

app = Flask('ping')

@app.route('/')
def home():
    return "Welcome to the Ping Pong server!"

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0' , port = 5001)