#libraries
import mediapipe as mp
import cv2
import numpy as np

def get_unique(c):
    temp_list = list(c)
    temp_set = set()
    for t in temp_list:
        temp_set.add(t[0])
        temp_set.add(t[1])
    return list(temp_set)


cap = cv2.VideoCapture("vid.mp4")
mp_face_mesh = mp.solutions.face_mesh
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#code for saving the video
size = (frame_width, frame_height)
res = cv2.VideoWriter('filename2.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

connections_irises = mp_face_mesh.FACEMESH_IRISES
irises_indices = get_unique(connections_irises)
connections_face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
face_oval_indices = get_unique(connections_face_oval)


with mp_face_mesh.FaceMesh(
    static_image_mode = False ,
    max_num_faces = 2 ,
    refine_landmarks = True ,
    min_detection_confidence = 0.5) as face_mesh:
    while(cap.isOpened()):
        ret , frame = cap.read()
        if ret == False:
            break
        results = face_mesh.process(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB))
        for face_landmark in results.multi_face_landmarks:
            lms = face_landmark.landmark
            d = {}
            for index in face_oval_indices:
                x = int(lms[index].x * frame.shape[1])
                y = int(lms[index].y * frame.shape[0])
                d[index] = (x , y)
            black = np.zeros(frame.shape).astype("uint8")
            for index in face_oval_indices:
                cv2.circle(black , (d[index][0] , d[index][1]) ,
                           2 , (0 , 255 , 0) , -1)
            for conn in list(connections_face_oval):
                cv2.line(black ,
                         (d[conn[0]][0] , d[conn[0]][1]) ,
                         (d[conn[1]][0] , d[conn[1]][1]) ,
                         (0 , 0 , 255) ,
                         1)
            res.write(black)
            if cv2.waitKey(5) & 0xFF == 27:
                break
res.release()
cap.release()
cv2.destroyAllWindows()








            







