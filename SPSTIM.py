import cv2
import numpy as np
import yaml, json
import sys, datetime
import firebase_admin
from firebase_admin import firestore
import requests, threading


firebase_credentials = firebase_admin.credentials.Certificate(r'smartpark.json')
firebase_admin.initialize_app(firebase_credentials)
firestore_database = firestore.client()

city = 'ifrane'

video_source_file = r"test.mov"
json_file = r'park_references.json'

do_nothing = lambda *args: None

parking_bounding_rectangles = []
parking_data = []
parking_contours = []
parking_bounding_rects = []
parking_mask = []

FREE_COLOR = (0, 0, 255)
OCCUPIED_COLOR = (0, 255, 0)

def initialize_database():
    for i in range (1, 11):
        document_reference = firestore_database.collection(u'parking').document(f'spot{i}')
        document_reference.set({
            u'status': 'Full',
            u'id': i
        })
    document_reference = firestore_database.collection(u'timestamp').document(u'time')
    document_reference.set({
        u'time': datetime.datetime.now().strftime("%c")
    })
    document_reference = firestore_database.collection(u'freespots').document(u'num')
    document_reference.set({
            u'num': 0
        })  
    doc_ref = firestore_database.collection(u'weather').document(u'decription')
    doc_ref.set({
            u'description': ''
        })

def set_weather():
    threading.Timer(600.0, set_weather).start()
    open_api = requests.get('http://api.openweathermap.org/data/2.5/weather?q='+city+'&appid=0b96d962faae72be191e2ce4ed1dcfe2')
    weather = json.loads(open_api.text)
    description = weather["weather"][0]["description"] 
    doc_ref = firestore_database.collection(u'weather').document(u'decription')
    doc_ref.update({
        u'description': description 
    })


def createTrackbars():
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('ClipLimit', 'Trackbars',  19, 255, do_nothing)
    cv2.createTrackbar('gridst', 'Trackbars',  10, 255, do_nothing)
    cv2.createTrackbar('gridfn', 'Trackbars',  2, 255, do_nothing)
    cv2.createTrackbar('cannylow', 'Trackbars',  0, 255, do_nothing)
    cv2.createTrackbar('cannyhigh', 'Trackbars',  0, 255, do_nothing)


def load_parking_data():
    global parking_data
    with open(json_file, 'r') as f:
        parking_data = json.load(f)
    for park_data in parking_data:
        bounding_rectangle_points = np.array(park_data['points'])
        bounding_rectangle = cv2.boundingRect(bounding_rectangle_points)
        points_shifted = bounding_rectangle_points.copy()
        points_shifted[:,0] = bounding_rectangle_points[:,0] - bounding_rectangle[0]
        points_shifted[:,1] = bounding_rectangle_points[:,1] - bounding_rectangle[1]
        parking_contours.append(bounding_rectangle_points)
        parking_bounding_rectangles.append(bounding_rectangle)
        mask = cv2.drawContours(np.zeros((bounding_rectangle[3], bounding_rectangle[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                            color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask==255
        parking_mask.append(mask)
        
def on_parking_status_change_listener(is_spot_empty, free_spots, spot_id):
    time = datetime.datetime.now().strftime("%c")
    doc_ref = firestore_database.collection(u'timestamp').document(u'time')
    doc_ref.update({
        u'time': time
    })

    if is_spot_empty:
        doc_ref = firestore_database.collection(u'parking').document(f'spot{spot_id+1}')
        doc_ref.update({
            u'status': 'Free',
        })
    else :
        doc_ref = firestore_database.collection(u'parking').document(f'spot{spot_id+1}')
        doc_ref.update({
            u'status': 'Full',
        })
    
    doc_ref = firestore_database.collection(u'freespots').document(u'num')
    doc_ref.update({
        u'num': free_spots
        })


def main():
    initialize_database()
    cap = cv2.VideoCapture(video_source_file)
    createTrackbars()
    set_weather()
    load_parking_data()
    parking_status = [False]*len(parking_data)
    parking_buffer = [None]*len(parking_data)
    while cap.isOpened():
        free_spots = 0
        video_current_position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Current position of the video file in seconds
        ret, frame = cap.read()
        if(ret is False):
            break
        output_frame = frame.copy()

        video_current_position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Current position of the video file in seconds
        video_current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES) # Index of the frame to be decoded/captured next

        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 3)
        grayed_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
        
        optimized_frame = cv2.createCLAHE(clipLimit=10, tileGridSize=(10, 3)).apply(grayed_frame)
        
        for spot_data in parking_data:
            spot_id = spot_data['id'] - 1

            bounding_rectangle = parking_bounding_rectangles[spot_id]
            x, y, w, h = bounding_rectangle

            roi = optimized_frame[y:y+h, x:x+w]

            laplacian = cv2.Laplacian(roi, cv2.CV_64F)
            mean = np.mean(np.abs(laplacian * parking_mask[spot_id]))
            is_spot_empty = mean < 4.2
            if(is_spot_empty):
                free_spots+=1
            if(spot_id==5):
                print(f'spot: {spot_id} is {is_spot_empty} has mean of {mean}')

            # If detected a change in parking status, save the current time & update database
            if is_spot_empty != parking_status[spot_id] and parking_buffer[spot_id]==None:
                parking_buffer[spot_id] = video_current_position
                on_parking_status_change_listener(is_spot_empty = is_spot_empty, free_spots = free_spots, spot_id = spot_id)

            # If status is still different than the one saved and counter is open
            elif is_spot_empty != parking_status[spot_id] and parking_buffer[spot_id]!=None:
                    if video_current_position - parking_buffer[spot_id] > 0.5:
                        parking_status[spot_id] = is_spot_empty
                        parking_buffer[spot_id] = None
            
            # If status is still same and counter is open                    
            elif is_spot_empty == parking_status[spot_id] and parking_buffer[spot_id]!=None:
                parking_buffer[spot_id] = None 
        
        for spot_data in parking_data:
            bounding_rectangle_points = np.array(spot_data['points'])
            spot_id = spot_data['id'] - 1

            if parking_status[spot_id]:
                color = OCCUPIED_COLOR
            else: 
                color = FREE_COLOR

            cv2.drawContours(output_frame, [bounding_rectangle_points], contourIdx=-1,
                             color=color, thickness=2, lineType=cv2.LINE_8)

            moments = cv2.moments(bounding_rectangle_points)        
            centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
            cv2.putText(output_frame, str(spot_id+1), (centroid[0]+1, centroid[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(output_frame, str(spot_id+1), (centroid[0]-1, centroid[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(output_frame, str(spot_id+1), (centroid[0]+1, centroid[1]-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(output_frame, str(spot_id+1), (centroid[0]-1, centroid[1]+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(output_frame, str(spot_id+1), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        cv2.imshow('output', output_frame)

        pressed_key = cv2.waitKey(1)
        if (pressed_key == 27 or pressed_key == ord('q')):
            break
        elif pressed_key == ord('c'):
            cv2.imwrite('frame%d.jpg' % video_current_frame, frame)
        elif pressed_key == ord('j'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_current_frame+2000)
        elif pressed_key == ord('b'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_current_frame-1000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()