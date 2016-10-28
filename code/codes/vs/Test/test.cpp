// This file is exploited for testing the face detection algorithm
//
// Code exploited by 2016 Feng Wang <feng.wff@gmail.com>
//
// This Source Code Form is subject to the terms of the BSD lisence.
#include <chrono>
#include <cstdlib>
#include <memory>
#include <Windows.h>

#include "boost/make_shared.hpp"
#include "TestFaceDetection.inc.h"

std::shared_ptr<caffe::CaffeBinding> kCaffeBinding = std::make_shared<caffe::CaffeBinding>();

using namespace FaceInception;

int CaptureDemo(CascadeCNN cascade) {
  VideoCapture cap(0);
  if (!cap.isOpened()) {
    return -1;
  }
  Mat frame;
  Mat edges;

  bool stop = false;
  Rect_<double> bakFaceRect;
  int bak_time = 0;
  while (!stop) {
    cap >> frame;
    if (frame.empty()) {
      cout << "cannot read from camera!" << endl;
      Sleep(100);
      continue;
    }
    vector<vector<Point2d>> points;
    std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
    double min_face_size = 40;
    auto result = cascade.GetDetection(frame, 12 / min_face_size, 0.7, true, 0.7, true, points);
    std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
    cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;
    for (int i = 0; i < result.size(); i++) {
      rectangle(frame, result[i].first, Scalar(255, 0, 0), 4);
      for (int p = 0; p < 5; p++) {
        circle(frame, points[i][p], 2, Scalar(0, 255, 255), -1);
      }
    }
    //resize(image, image, Size(0, 0), 0.5, 0.5);
    imshow("capture", frame);
    waitKey(1);
  }
}
void ScanList(string root_folder, CascadeCNN cascade) {
  std::ifstream infile(root_folder.c_str());
  string filename;
  char this_line[65536];
  int label;

  while (!infile.eof()) {
    memset(this_line, 0, sizeof(this_line));
    infile.getline(this_line, 65536);
    if (strlen(this_line) == 0) continue;
    this_line[strlen(this_line) - 1] = '\0';
    std::stringstream stream;
    stream << this_line;
    stream >> filename >> label;
    cout << filename << endl;
    try {
      Mat image = imread(filename);
      vector<vector<Point2d>> points;
      auto result = cascade.GetDetection(image, 1, 0.995, true, 0.3, true, points);
      for (int i = 0; i < result.size(); i++) {
        rectangle(image, result[i].first, Scalar(255, 0, 0), 4);
        for (int p = 0; p < 5; p++) {
          circle(image, points[i][p], 2, Scalar(0, 255, 255), -1);
        }
      }
      //resize(image, image, Size(0, 0), 0.5, 0.5);
      imshow("boxes", image);
      waitKey(0);
    }
    catch (std::exception e) {}
  }
}


int main(int argc, char* argv[])
{
  //CascadeCNN cascade("G:\\WIDER\\face_detection\\bak3\\cascade_12_memory_nobn1.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade12-_iter_490000.caffemodel",
  //                   "G:\\WIDER\\face_detection\\bak3\\cascade_24_memory_full.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade24-_iter_145000.caffemodel",
  //                   "G:\\WIDER\\face_detection\\bak3\\cascade_48_memory_full.prototxt", "G:\\WIDER\\face_detection\\bak3\\cascade48-_iter_225000.caffemodel");
  string model_folder = "D:\\face project\\MTCNN_face_detection_alignment\\code\\codes\\MTCNNv2\\model\\";
  CascadeCNN cascade(model_folder+"det1-memory.prototxt", model_folder + "det1.caffemodel",
                     model_folder + "det2-memory.prototxt", model_folder + "det2.caffemodel",
                     model_folder + "det3-memory.prototxt", model_folder + "det3.caffemodel",
                     model_folder + "det4-memory.prototxt", model_folder + "det4.caffemodel",
                     0);
  //CaptureDemo(cascade);

  double min_face_size = 12;

  //ScanList("H:\\lfw\\list.txt", cascade);
  //Mat image = imread("D:\\face project\\WIDER\\face_detection\\oscar1.jpg");
  //Mat image = imread("G:\\WIDER\\face_detection\\pack\\1[00_00_26][20160819-181452-0].BMP");
  Mat image = imread("D:\\face project\\FDDB\\2003/01/13/big/img_1087.jpg");
  //Mat image = imread("D:\\face project\\FDDB\\2003/01/13/big/img_1087.bmp");
  cout << image.cols<<","<<image.rows << endl;
  vector<vector<Point2d>> points;
  std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
  auto result = cascade.GetDetection(image, 12.0 / min_face_size, 0.7, true, 0.7, true, points);
  std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
  cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;

  cout << "===========================================================" << endl;
  points.clear();//The first run is slow because it need to allocate memory.
  p0 = std::chrono::system_clock::now();
  result = cascade.GetDetection(image, 12.0 / min_face_size, 0.7, true, 0.7, true, points);
  p1 = std::chrono::system_clock::now();
  cout << "detection time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;

  for (int i = 0; i < result.size(); i++) {
    //cout << "face box:" << result[i].first << " confidence:" << result[i].second << endl;
    rectangle(image, result[i].first, Scalar(255, 0, 0), 2);
    if (points.size() >= i + 1) {
      for (int p = 0; p < 5; p++) {
        circle(image, points[i][p], 2, Scalar(0, 255, 255), -1);
      }
    }
  }
  //resize(image, image, Size(0, 0), 0.25, 0.25);
  imshow("final", image);
  waitKey(0);
  //imwrite("output.jpg", image);
  cascade.TestFDDBPrecision("D:\\face project\\FDDB\\", true, true);
  system("pause");
	return 0;
}

