import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from Emotion.utils.datasets import get_labels
from Emotion.utils.inference import detect_faces
from Emotion.utils.inference import draw_text
from Emotion.utils.inference import draw_bounding_box
from Emotion.utils.inference import apply_offsets
from Emotion.utils.inference import load_detection_model
from Emotion.utils.preprocessor import preprocess_input
import time



def negatives(name):
    if ('No' in name or 'no' in name or 'naah' in name or 'nope' in name or 'dont' in name):
        return True
    else:
        return False

def positives(name):
    if ('Yes' in name or 'yes' in name or 'yup' in name or 'Yeah' in name or 'yeah' in name):
        return True
    else:
        return False

def check_sent(name):
    if('name' in name or 'is' in name):
            return True

def check_sent_name(name):
    return name[(name.index('is')+1)].split()


def emotion_detect(n, depresso_bot):
    USE_WEBCAM = True # If false, loads video file source

    avg_sad=0
    avg_happy=0
    avg_neutral=0
    avg_angry=0
    avg_surprised=0

    # parameters for loading data and images
    emotion_model_path = 'Emotion/models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_cascade = cv2.CascadeClassifier('Emotion/models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    # starting video streaming

    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)

    # Select video or webcam feed
    cap = None
    if (USE_WEBCAM == True):
        cap = cv2.VideoCapture(0) # Webcam source
    else:
        cap = cv2.VideoCapture('Emotion/demo/dinner.mp4') # Video file source
    c=0
    nc=0
    print("Enter the length of the session in minutes")
    length = int(input())
    seconds = length * 60
    a=time.time()
    b=time.time()
    name_fin=""
    while (cap.isOpened() and b-a<=seconds): # True:

        ret, bgr_image = cap.read()
        if(n%11==10):
            """
            Insert chatterbot code here


            """
            print("Enter response: ",end="")
            msg=input()
            c+=1
            print(depresso_bot.get_response(msg))
            """ Asking for the Username """
            if(c%3==0 and nc==0):

                print('May I know your name? :)')
                name=input().split()

                #Checking if the user says 'My name is . . .'
                if check_sent(name):
                    name=(check_sent_name(name))
                    nc=1
                    print('Hello ',name[0])
                    name_fin=name[0]
                elif negatives(name):
                    print('Umm okay :/')
                    nc=1
                elif positives(name):
                    print('Please enter your name:')
                    name=input().split()
                    if check_sent(name):
                        name=check_sent_name(name)
                    else:
                        name=input().split()
                    print('Hello ', name[0])
                    nc=1
                    name_fin=name[0]


                elif(len(name)>2):
                    while True:
                        print('Are you sure that is your name?')
                        check_name = input()
                        if positives(check_name):
                            print('Hello ', name[0])
                            name_fin=name[0]
                            nc=1
                            break
                        elif negatives(check_name):
                            print('Would you mind telling your name again?')
                            check_name2=input().split()
                            if check_sent(check_name2):
                                name=(check_sent_name(name))
                                print('Hello ',name[0])
                                name_fin=name[0]
                                nc=1
                                break
                            elif positives(check_name2):
                                print('Please tell your name')
                                name=input().split()
                                if check_sent(name):
                                    name=(check_sent_name(name))
                                print('Hello ',name[0])
                                name_fin=name[0]
                                nc=1
                                break
                            elif negatives(check_name2):
                                print('Okay nevermind :p')
                                break
                            else:
                                name=check_name2
                                print('Hello ',check_name2[0])
                                name_fin=check_name2[0]
                                nc=1
                                break
                else:
                    print("Hello ", name[0])
                    name_fin=name[0]
                    nc=1

            """
            Chatterbot code ends here
            """
        n+=1
        #bgr_image = video_capture.read()[1]

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)



        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
    			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
                avg_angry+=1

            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
                avg_sad+=1
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
                avg_happy+=1
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
                avg_surprised+=1
            else:
                color = emotion_probability * np.asarray((0, 255, 0))
                avg_neutral+=1

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        b=time.time()

    cap.release()
    cv2.destroyAllWindows()
    print("\n")
    print("Session over")
    print("Analysis of emotions: " )
    sum_avg = avg_sad+avg_angry+avg_happy+avg_happy+avg_neutral+avg_surprised

    phappy= avg_happy/sum_avg*100
    psad = avg_sad/sum_avg*100
    pangry = avg_angry/sum_avg*100
    psurprised = avg_surprised/sum_avg*100
    pneutral = avg_neutral/sum_avg*100
    print("Percentage Happy: ",phappy)
    print("Percentage Sad: ",psad)
    print("Percentage Angry: ",pangry)
    print("Percentage Surprised: ",psurprised)
    print("Percentage Neutral: ",pneutral)
    print()
    print("Analysis from chat: " )
    name_fin = name_fin.replace(" ",'')
    print("Name of the candidate: ",name_fin)
