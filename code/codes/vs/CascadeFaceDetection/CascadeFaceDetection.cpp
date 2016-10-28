#include "CascadeFaceDetection.h"
#include "CaffeBinding.h"
#include "../Test/util/BoundingBox.inc.h"
#include "../Test/TestFaceDetection.inc.h"
#include "boost/make_shared.hpp"

using namespace std;
boost::shared_ptr<caffe::CaffeBinding> kCaffeBinding = boost::make_shared<caffe::CaffeBinding>();

namespace FaceInception {
  CascadeCNN cascade;
  CascadeFaceDetection::CascadeFaceDetection() {
    cout << "Please specify the net models." << endl;
  }

  CascadeFaceDetection::CascadeFaceDetection(string net12_definition, string net12_weights,
                                             string net24_definition, string net24_weights,
                                             string net48_definition, string net48_weights) {
    cascade = CascadeCNN(net12_definition, net12_weights,
                         net24_definition, net24_weights,
                         net48_definition, net48_weights);
  }

  FaceInformation CascadeFaceDetection::Predict(cv::Mat& input_image, double min_confidence) {
    FaceInformation result;
    auto rect_and_score = cascade.GetDetection(input_image, 0.5, min_confidence, true, 0.3, true, result.points);
    for (auto& rs : rect_and_score) {
      result.boundingbox.push_back(rs.first);
      result.confidence.push_back(rs.second);
    }
    return result;
  }

  CascadeFaceDetection::~CascadeFaceDetection() {

  }
}