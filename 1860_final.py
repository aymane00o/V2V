import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
import cv2
import numpy as np
import os
import serial
import logging
from RPLCD.i2c import CharLCD
import paho.mqtt.client as mqtt
import threading

# Define the MQTT broker address and port
broker_address = "192.168.83.246"
broker_port = 1883
topic = "home/V2V"

# Initialize MQTT client
client = mqtt.Client()
client.connect(broker_address, broker_port)

# Setup logging
logging.basicConfig(level=logging.INFO)

# LCD setup
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)

# GPIO Pins
PIN_FORWARD, PIN_REVERSE, PIN_LEFT, PIN_RIGHT, PIN_ENABLE = 23, 24, 17, 22, 18

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
for pin in [PIN_FORWARD, PIN_REVERSE, PIN_LEFT, PIN_RIGHT, PIN_ENABLE]:
    GPIO.setup(pin, GPIO.OUT)
GPIO.output(PIN_ENABLE, GPIO.HIGH)

# Initialize PWM
def initialize_pwm(pins, frequency=100):
    return {pin: GPIO.PWM(pin, frequency) for pin in pins}

pwm_pins = initialize_pwm([PIN_FORWARD, PIN_REVERSE, PIN_LEFT, PIN_RIGHT])
for pwm in pwm_pins.values():
    pwm.start(0)

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()
time.sleep(0.1)

# Frame center
frame_center = 320

# Serial setup for obstacle detection
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

# File paths for models and cascades
model_files = {
    "yolo_cfg": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/yolov3-tiny.cfg",
    "yolo_weights": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/yolov3-tiny.weights",
    "coco_names": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/coco.names",
    "speed_limit": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/Speedlimit_HAAR_ 17Stages.xml",
    "bumper_sign": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/bumpersign_classifier_haar.xml",
    "traffic_light": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/TrafficLight_HAAR_16Stages.xml",
    "traffic_sign": "/home/aymane/Downloads/cascade.xml",
    "yield_sign": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/yieldsign12Stages.xml",
    "turn_left": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/turnLeft_ahead (1).xml",
    "turn_right": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/turnRight_ahead.xml",
    "stop_sign": "/home/aymane/Desktop/hahahahhaah2/code_mixe_detecting/stop_sign_classifier_2.xml"
}

# Check if files exist
def check_files(files):
    for path in files.values():
        if not os.path.exists(path):
            logging.error(f"Missing file: {path}")
            exit(1)

check_files(model_files)

# Load YOLO model and classes
net = cv2.dnn.readNet(model_files["yolo_weights"], model_files["yolo_cfg"])
with open(model_files["coco_names"], "r") as f:
    classes = [line.strip() for line in f.readlines()]
output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load cascade classifiers
speed_limit_cascade = cv2.CascadeClassifier(model_files["speed_limit"])
bumper_sign_cascade = cv2.CascadeClassifier(model_files["bumper_sign"])
traffic_light_cascade = cv2.CascadeClassifier(model_files["traffic_light"])
traffic_sign_cascade = cv2.CascadeClassifier(model_files["traffic_sign"])
yield_sign_cascade = cv2.CascadeClassifier(model_files["yield_sign"])
turn_left_cascade = cv2.CascadeClassifier(model_files["turn_left"])
turn_right_cascade = cv2.CascadeClassifier(model_files["turn_right"])
stop_sign_cascade = cv2.CascadeClassifier(model_files["stop_sign"])

# Motor control functions
def stop_motors():
    logging.info("Stopping motors")
    for pwm in pwm_pins.values():
        pwm.ChangeDutyCycle(0)

def move_motor(motor, speed):
    for pin in pwm_pins:
        pwm_pins[pin].ChangeDutyCycle(speed if pin == motor else 0)

def forward(speed=10):
    move_motor(PIN_FORWARD, speed)

def reverse(speed=10):
    move_motor(PIN_REVERSE, speed)
    
def steer_left(forward_speed=10, steer_speed=10):
    pwm_pins[PIN_FORWARD].ChangeDutyCycle(forward_speed)
    pwm_pins[PIN_LEFT].ChangeDutyCycle(steer_speed)

def steer_right(forward_speed=10, steer_speed=10):
    pwm_pins[PIN_FORWARD].ChangeDutyCycle(forward_speed)
    pwm_pins[PIN_RIGHT].ChangeDutyCycle(steer_speed)

# Image processing functions
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    mask = np.zeros_like(binary)
    height, width = binary.shape
    polygon = np.array([[(0, height), (width, height), (width, int(height * 0.6)), (0, int(height * 0.6))]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.findContours(cv2.bitwise_and(binary, mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

# Traffic sign detection functions
def detect_speed_limit_signs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in speed_limit_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Speed Limit", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        client.publish(topic, "Speed Limit")
        forward(10)
        lcd.write_string('Speed Limit!!')
        time.sleep(2)
        lcd.clear()
    return frame
def detect_stop_signs(frame):
    global stop_sign_detected
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stop_signs = stop_sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in stop_signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Stop Sign", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        client.publish(topic, "Stop Sign")
        lcd.write_string('Stop Sign')
        stop_motors()
        stop_sign_detected = True
        time.sleep(2)
        lcd.clear()
        time.sleep(5)
        forward(50)
    return frame

def detect_bumper_signs(frame):
    global bumper_detected
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bumpers = bumper_sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), maxSize=(200, 200))

    for (x, y, w, h) in bumpers:
        if 50 <= w <= 200 and 50 <= h <= 200 and 0.6 <= w / h <= 1.4:
            if is_triangular(frame[y:y + h, x:x + w]):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Bumper sign", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                bumper_detected = True
                client.publish(topic, "Bumper")
                lcd.write_string('Bumper Ahead')
                time.sleep(2)
                lcd.clear()
    return frame

def detect_traffic_signs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    signs = traffic_sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Traffic Sign", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        client.publish(topic, "Traffic Sign")
        lcd.write_string('passage')
        time.sleep(2)
        lcd.clear()
    return frame

def detect_yield_signs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    yield_signs = yield_sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in yield_signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, "Yield Sign", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        client.publish(topic, "Yield Sign")
        lcd.write_string('Yield Sign')
        forward(10)
        time.sleep(2)
        lcd.clear()
    return frame

def detect_turn_left_signs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    turn_left_signs = turn_left_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in turn_left_signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Turn Left Ahead", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        client.publish(topic, "Turn Left Ahead")
        lcd.write_string('Turn Left')
        steer_left()
        time.sleep(2)
        lcd.clear()
    return frame

def detect_turn_right_signs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    turn_right_signs = turn_right_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in turn_right_signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "Turn Right Ahead", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        client.publish(topic, "Turn Right Ahead")
        lcd.write_string('Turn Right')
        steer_right()
        time.sleep(2)
        lcd.clear()
    return frame

def detect_traffic_lights(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lights = traffic_light_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in lights:
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Define color ranges for red, green, and orange lights
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.add(red_mask1, red_mask2)

        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

        if cv2.countNonZero(red_mask) > max(cv2.countNonZero(green_mask), cv2.countNonZero(orange_mask)):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Red Light", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            client.publish(topic, "Red Light")
            lcd.write_string('Red Light')
            stop_motors()
            time.sleep(10)
        elif cv2.countNonZero(green_mask) > max(cv2.countNonZero(red_mask), cv2.countNonZero(orange_mask)):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Green Light", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            client.publish(topic, "Green Light")
            lcd.write_string('Green Light')
            forward(20)
        elif cv2.countNonZero(orange_mask) > max(cv2.countNonZero(red_mask), cv2.countNonZero(green_mask)):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(frame, "Orange Light", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            client.publish(topic, "Orange Light")
            lcd.write_string('Orange Light')
        time.sleep(2)
        lcd.clear()
    return frame

# Function to check if a contour is roughly triangular
def is_triangular(region):
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        approx = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)
        if len(approx) == 3:
            return True
    return False

def detect_obstacle():
    try:
        payload = ser.readline().decode('utf-8').rstrip()
        if payload == "Obstacle":
            logging.info("Obstacle detected")
            client.publish(topic, "Obstacle")
            stop_motors()
            lcd.write_string('Attention')
            time.sleep(2)
            lcd.clear()
            return True
    except Exception as e:
        logging.error(f"Error in detecting obstacle: {e}")
    return False

def obstacle_detection_thread(obstacle_event):
    while True:
        if detect_obstacle():
            obstacle_event.set()
           
            
        else:
            obstacle_event.clear()
            
        time.sleep(0.1)

# Main loop
def main():
    forward(50)
    obstacle_event = threading.Event()
    
    obstacle_thread = threading.Thread(target=obstacle_detection_thread, args=(obstacle_event,))
    obstacle_thread.daemon = True
    obstacle_thread.start()

    try:
        forward(50)
        while True:
            if obstacle_event.is_set(): 
                time.sleep(1)
                continue
            
           
            frame = cv2.resize(picam2.capture_array(), (320, 240))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if frame.shape[2] == 4 else frame

            contours = process_image(frame)

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (210, 210), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids, confidences, boxes = [], [], []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        x, y, w, h = map(int, [detection[0] * frame.shape[1] - detection[2] * frame.shape[1] / 2,
                                               detection[1] * frame.shape[0] - detection[3] * frame.shape[0] / 2,
                                               detection[2] * frame.shape[1],
                                               detection[3] * frame.shape[0]])
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            for i in cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4):
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                color = colors[class_ids[i]]
            
                

            frame = detect_speed_limit_signs(frame)
            frame = detect_bumper_signs(frame)
            frame = detect_traffic_lights(frame)
            frame = detect_traffic_signs(frame)
            frame = detect_yield_signs(frame)
            frame = detect_turn_left_signs(frame)
            frame = detect_turn_right_signs(frame)
            frame = detect_stop_signs(frame)

           

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            

    except Exception as e:
        logging.error(f"Error in main loop: {e}")

    finally:
        cv2.destroyAllWindows()
        picam2.stop()
        for pwm in pwm_pins.values():
            pwm.stop()
        GPIO.output(PIN_ENABLE, GPIO.LOW)
        GPIO.cleanup()

if __name__ == "__main__":
    main()

