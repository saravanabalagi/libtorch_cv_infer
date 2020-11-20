#include <iostream>
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat readImg(string img_path) {
  Mat img = imread(img_path);
  cvtColor(img, img, COLOR_BGR2RGB);
  return img;
}

Mat resizeNormalizeImg(Mat img) {
  int img_h = 224, img_w = 224;
  if(img.rows == img_h && img.cols == img_w)
    return img;
  Mat img_resized = Mat(img_h, img_w, CV_8UC3);
  resize(img, img_resized, img_resized.size(), INTER_AREA);
  img_resized.convertTo(img_resized, CV_32F);
  img_resized -= Scalar(70.370258669401, 74.122797553143, 73.178495195211);   // Subtract mean
  img_resized /= Scalar(51.644749102296, 51.515240911425, 52.277554279530);   // Divide Std
  return img_resized;
}

torch::Tensor getInputTensor(Mat img) {
  torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, at::kByte);
  tensor = tensor.permute(torch::IntList({0, 3, 1, 2}));    // Channel first ordering for torch
  tensor = tensor.toType(torch::kFloat32);                  // Covert uint8 to float32
  tensor = tensor.contiguous();                             // Make contiguous for torch.view()
  return tensor;
}

int main(int argc, char **argv) {

  if (!(argc == 3 || argc == 4)) {
    cerr << "usage: predict <path-to-image> <model_path> [<use_gpu>] \n";
    return -1;
  }
  
  // Parse args
  string img_path = argv[1];
  string model_path = argv[2];
  string usegpu_str = (argc == 3) ? "true" : argv[3];
  bool usegpu = (usegpu_str == "true") ? true : false;

  // Build model
  torch::Device device(usegpu ? torch::kCUDA : torch::kCPU);
  torch::Device cpu(torch::kCPU);
  torch::jit::script::Module model = torch::jit::load(model_path);
  model.to(device);
  model.eval();

  // Resize image if necessary
  int img_height = 224;
  int img_width = 224;
  Mat img = readImg(img_path);
  img = resizeNormalizeImg(img);
  torch::Tensor inputTensor = getInputTensor(img);
  cout << "input shape: " << inputTensor.sizes() << endl;

  // Inference
  torch::NoGradGuard no_grad;
  torch::Tensor output = model.forward({inputTensor.to(device)}).toTensor();    // If model has 1 output
  cout << "output shape: " << output.sizes() << endl;

  return 0;
}
