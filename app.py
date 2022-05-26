import csv
import json
import time
import pandas as pd
from gevent.pywsgi import WSGIServer
from gevent import monkey; monkey.patch_all()
from flask import Flask, Response, render_template, stream_with_context

app = Flask(__name__)

@app.route("/")
def render_index():
    return render_template("index.html")

@app.route("/listen")
def listen():

  def respond_to_client():
    while True:
      sample_data = pd.read_csv("Supporting Material/Average.csv", header = None, quoting=csv.QUOTE_NONE, encoding='utf-8')
   
      if(1):
        if( int(sample_data.iloc[-1][0]) > int(sample_data.iloc[-1][1]) and
            int(sample_data.iloc[-1][0]) > int(sample_data.iloc[-1][2]) and
            int(sample_data.iloc[-1][0]) > int(sample_data.iloc[-1][3]) and
            int(sample_data.iloc[-1][0]) > int(sample_data.iloc[-1][4]) and
            int(sample_data.iloc[-1][0]) > int(sample_data.iloc[-1][5])):
            color = "rgb(100, 0, 0)"
        if( int(sample_data.iloc[-1][1]) > int(sample_data.iloc[-1][0]) and
            int(sample_data.iloc[-1][1]) > int(sample_data.iloc[-1][2]) and
            int(sample_data.iloc[-1][1]) > int(sample_data.iloc[-1][3]) and
            int(sample_data.iloc[-1][1]) > int(sample_data.iloc[-1][4]) and
            int(sample_data.iloc[-1][1]) > int(sample_data.iloc[-1][5])):
            color = "rgb(100, 20, 70)"
        if( int(sample_data.iloc[-1][2]) > int(sample_data.iloc[-1][1]) and
            int(sample_data.iloc[-1][2]) > int(sample_data.iloc[-1][0]) and
            int(sample_data.iloc[-1][2]) > int(sample_data.iloc[-1][3]) and
            int(sample_data.iloc[-1][2]) > int(sample_data.iloc[-1][4]) and
            int(sample_data.iloc[-1][2]) > int(sample_data.iloc[-1][5])):
            color = "rgb(0, 0, 0)"
        if( int(sample_data.iloc[-1][3]) > int(sample_data.iloc[-1][1]) and
            int(sample_data.iloc[-1][3]) > int(sample_data.iloc[-1][2]) and
            int(sample_data.iloc[-1][3]) > int(sample_data.iloc[-1][0]) and
            int(sample_data.iloc[-1][3]) > int(sample_data.iloc[-1][4]) and
            int(sample_data.iloc[-1][3]) > int(sample_data.iloc[-1][5])):
            color = "rgb(0, 100, 0)"
        if( int(sample_data.iloc[-1][4]) > int(sample_data.iloc[-1][1]) and
            int(sample_data.iloc[-1][4]) > int(sample_data.iloc[-1][2]) and
            int(sample_data.iloc[-1][4]) > int(sample_data.iloc[-1][3]) and
            int(sample_data.iloc[-1][4]) > int(sample_data.iloc[-1][0]) and
            int(sample_data.iloc[-1][4]) > int(sample_data.iloc[-1][5])):
            color = "rgb(0, 0, 100)"
        if( int(sample_data.iloc[-1][5]) > int(sample_data.iloc[-1][1]) and
            int(sample_data.iloc[-1][5]) > int(sample_data.iloc[-1][2]) and
            int(sample_data.iloc[-1][5]) > int(sample_data.iloc[-1][3]) and
            int(sample_data.iloc[-1][5]) > int(sample_data.iloc[-1][4]) and
            int(sample_data.iloc[-1][5]) > int(sample_data.iloc[-1][0])):
            color = "rgb(90, 90, 0)"
        _data = json.dumps({"color":color, "c1":str(sample_data.iloc[-1][0]), "c2":str(sample_data.iloc[-1][1]), "c3":str(sample_data.iloc[-1][2]), "c4":str(sample_data.iloc[-1][3]), "c5":str(sample_data.iloc[-1][4]), "c6":str(sample_data.iloc[-1][5]), "c7":str(sample_data.iloc[-1][6])})
        yield f"id: 1\ndata: {_data}\nevent: online\n\n"
  return Response(respond_to_client(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(port=80, debug=True)
    http_server = WSGIServer(("localhost", 80), app)
    http_server.serve_forever()
