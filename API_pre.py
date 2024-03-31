from flask import Flask , request, jsonify
import numpy as np
import cv2
import base64
from matplotlib import pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload():
        try:
            print ("เข้า")
            uploaded_file = request.files['file']
            print (uploaded_file)
            if uploaded_file.filename == '':
                return jsonify({"success": False, "message": str(e)})
            else:
                #read image ต้องเปลี่ยนจากไบนารี่มาเป็นสติง
                image_nparr = np.asarray(bytearray(uploaded_file.read()), dtype="uint8")
                img = cv2.imdecode(image_nparr, cv2.IMREAD_COLOR)
                blurimg = cv2.GaussianBlur(img, (5, 5), 0)
                gray = cv2.cvtColor(blurimg, cv2.COLOR_BGR2GRAY)

                th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
                thinv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15)

                def region_growing(img, seed):
                    h, w = img.shape
                    mask = np.zeros_like(img, dtype=np.uint8)

                    queue = []
                    queue.append(seed)

                    connectivity = [(0, 1), (0, -1), (1, 0), (-1, 0)]

                    while len(queue) > 0:
                        x, y = queue.pop()

                        if mask[x, y] == 0:
                            mask[x, y] = 255

                            for dx, dy in connectivity:
                                nx, ny = x + dx, y + dy

                                if nx >= 0 and nx < h and ny >= 0 and ny < w:
                                    if abs(int(img[nx, ny]) - int(img[x, y])) <= 254:
                                        queue.append((nx, ny))

                    return mask

                seed_x, seed_y = 100, 100

                region = region_growing(th, (seed_x, seed_y))

                dist = cv2.distanceTransform(thinv, cv2.DIST_L1, 3)

                kernel = np.ones((3, 3), np.uint8)
                erosion = cv2.erode(dist, kernel, iterations=1)
                erosion = cv2.convertScaleAbs(erosion)

                # contours, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours, hierarchy = cv2.findContours(erosion.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_with_contours = img.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX

                valid_contours = 0
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)

                    if w > 1 and h > 1:
                        # Replace rectangle with a dot
                        center_x = x + w // 2
                        center_y = y + h // 2
                        cv2.circle(img_with_contours, (center_x, center_y), 2, (0, 255, 0), -1)
                        # cv2.putText(img_with_contours, str(valid_contours + 1), (x, y - 10), font, 0.25, (0, 0, 255), 1)
                        valid_contours += 1

                print("Birds in image =", valid_contours)
                
                _, img_count_encoded = cv2.imencode('.jpg', img_with_contours)
                _, img_original_encoded = cv2.imencode('.jpg', img)

                bird_count = (str(valid_contours))
                
                img_original = base64.b64encode(img_original_encoded.tobytes()).decode('utf-8')
                img_count = base64.b64encode(img_count_encoded.tobytes()).decode('utf-8')

                data_list = {
                    "success": True,
                    "image_original": img_original,
                    "image_count": img_count,
                    "bird_count": bird_count,
                    "message": 'upload image success'
                }

                return jsonify(data_list) ,200

        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    
if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='10.199.121.49', port=8080)
    #ใช้ ipv4 

