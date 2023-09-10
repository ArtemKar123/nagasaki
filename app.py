from flask import Flask, request
import cv2
import numpy as np
import json
import os
from recognition.Processor import Processor

host = os.environ.get('HOST', '0.0.0.0')
port = int(os.environ.get('PORT', 5001))

processor = Processor(os.path.join(os.getcwd(), 'recognition/weights/detect.pt'),
                      os.path.join(os.getcwd(), 'recognition/weights/classify.pt'))

app = Flask(__name__)


@app.route('/tiles', methods=['POST'])
def process_tiles():
    r = request
    nparr = np.frombuffer(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = {"tiles": processor.find_tiles(img)}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run(host=host, port=port)
