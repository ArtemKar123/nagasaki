import random
import time

from flask import Flask, request
import cv2
import numpy as np
import json
import os
from recognition.Processor import Processor
import base64

host = os.environ.get('HOST', '0.0.0.0')
port = int(os.environ.get('PORT', 5001))

processor = Processor(os.path.join(os.getcwd(), 'recognition/weights/detect2.pt'),
                      os.path.join(os.getcwd(), 'recognition/weights/classify4.pt'),
                      save=True)

app = Flask(__name__)


@app.route('/tiles', methods=['POST'])
def process_tiles():
    data = request.get_json()
    img_base64 = data['image']

    # Convert base64 image to numpy array
    nparr = np.fromstring(base64.b64decode(img_base64), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite('recognition/tmp/full/' + str(time.time()) + '.jpg', img)

    result = {"tiles": processor.get_tiles(img)}
    print(result)
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run(host=host, port=port)
