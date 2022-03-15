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


img = cv2.imread("face_pot.jpg")
img = cv2.resize(img , (600 , 600))
mp_face_mesh = mp.solutions.face_mesh

connections_irises = mp_face_mesh.FACEMESH_IRISES
irises_indices = get_unique(connections_irises)
connections_face_oval = mp_face_mesh.FACEMESH_FACE_OVAL
face_oval_indices = get_unique(connections_face_oval)

black = np.zeros(img.shape).astype("uint8")

with mp_face_mesh.FaceMesh(
    static_image_mode = False ,
    max_num_faces = 2 ,
    refine_landmarks = True ,
    min_detection_confidence = 0.5) as face_mesh:

    annotated_image = img.copy()
    results = face_mesh.process(cv2.cvtColor(img , cv2.COLOR_BGR2RGB))
    for face_landmark in results.multi_face_landmarks:
        lms = face_landmark.landmark
        d = {}
        for index in irises_indices:
            x = int(lms[index].x * img.shape[1])
            y = int(lms[index].y * img.shape[0])
            d[index] = (x , y)
        for index in irises_indices:
            cv2.circle(black , (d[index][0] , d[index][1]) ,
                       2 , (0 , 255 , 0) , -1)
        for conn in list(connections_irises):
            cv2.line(black ,
                     (d[conn[0]][0] , d[conn[0]][1]) ,
                     (d[conn[1]][0] , d[conn[1]][1]) ,
                     (0 , 0 , 255) ,
                     1)
cv2.imshow("final" , black)







