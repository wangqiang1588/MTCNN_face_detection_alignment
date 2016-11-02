import cv2
import sys;
import os;
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../x64/Release"))
import CascadeFaceDetection

model_folder = "D:/face project/MTCNN_face_detection_alignment/code/codes/MTCNNv2/model/"

CascadeCNN = CascadeFaceDetection.CascadeCNN(model_folder + "det1-memory.prototxt", model_folder + "det1.caffemodel",
                     model_folder + "det2-memory.prototxt", model_folder + "det2.caffemodel",
                     model_folder + "det3-memory.prototxt", model_folder + "det3.caffemodel",
                     model_folder + "det4-memory.prototxt", model_folder + "det4.caffemodel",
                     0)
I = cv2.imread("c:/lena.png")
result = CascadeCNN.Predict(I, 0.96)
