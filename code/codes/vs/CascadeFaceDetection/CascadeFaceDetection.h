#pragma once

#ifdef CASCADEDETECTION_EXPORTS
#define CASCADE_DLL __declspec(dllexport)
#else
#define CASCADE_DLL __declspec(dllimport)
#endif

#include <opencv2\opencv.hpp>

namespace FaceInception {
  struct FaceInformation {
    std::vector<cv::Rect2d> boundingbox;
    std::vector<float> confidence;
    std::vector<std::vector<cv::Point2d>> points;
  };
  class CASCADE_DLL CascadeFaceDetection {
  public:
    CascadeFaceDetection();
    CascadeFaceDetection(std::string net12_definition, std::string net12_weights,
                         std::string net24_definition, std::string net24_weights,
                         std::string net48_definition, std::string net48_weights);
    FaceInformation Predict(cv::Mat& input_image, double min_confidence = 0.995);
    ~CascadeFaceDetection();
  };
}