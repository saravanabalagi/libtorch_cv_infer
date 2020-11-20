#include <iostream>
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


int main(int argc, char **argv) {

  if (!(argc == 3 || argc == 4)) {
    std::cerr << "usage: predict <path-to-image> <model_path> [<use_gpu>] \n";
    return -1;
  }
  
  // Parse args
  std::string img_path = argv[1];
  std::string model_path = argv[2];
  std::string usegpu_str = (argc == 3) ? "true" : argv[3];
  bool usegpu = (usegpu_str == "true") ? true : false;

  // Resize image if necessary
  int img_height = 224;
  int img_width = 224;
  cv::Mat img_orig = cv::imread(img_path);
  cv::Mat img;
  if(img_orig.rows != img_height || img_orig.cols != img_width) {
    img = cv::Mat(img_height, img_width, CV_8UC3);
    cv::resize(img_orig, img, img.size(), cv::INTER_AREA);
  } else img = img_orig;

  // Build model
  torch::Device device(usegpu ? torch::kCUDA : torch::kCPU);
  torch::Device cpu(torch::kCPU);
  torch::jit::script::Module model = torch::jit::load(model_path);
  model.to(device);
  model.eval();

  // Build input
  torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, at::kByte);
  tensor = tensor.permute(torch::IntList({0, 3, 1, 2}));    // Channel first ordering for torch
  tensor = tensor.toType(torch::kFloat32);                  // Covert uint8 to float32
  tensor = tensor.contiguous();                             // Make contiguous for torch.view()
  tensor = tensor.to(device);                               // Load in GPU
  std::cout << "input shape: " << tensor.sizes() << std::endl;
  torch::NoGradGuard no_grad;

  // Inference
  torch::Tensor output = model.forward({tensor}).toTensor();    // If model has 1 output
  std::cout << "output shape: " << output.sizes() << std::endl;

  // Debug whole model
  // auto output = model.forward({tensor});
  // // torch::Tensor output = model.sample_latent({tensor}).toTensor();    // If model has 1 output
  // torch::Tensor mu = output.toTuple()->elements()[0].toTensor();         // Model has 2 output, mu and logvar
  // mu = mu.to(cpu);
  // std::cout << "output shape: " << mu.sizes() << std::endl;

  // cv::Mat desc = cv::Mat(1, mu.sizes()[1], CV_32F, mu.data_ptr());
  // std::cout << "output shape: " << desc.size() << std::endl;

  return 0;
}

void normalize_img(cv::Mat img) {
  int img_h = 224, img_w = 224;
  if(img.rows != img_h || img.cols != img_w)
    return;
  img = cv::Mat(img_h, img_w, CV_8UC3);
  cv::resize(img, img, img.size(), cv::INTER_AREA);
}

cv::Mat read_img(std::string img_path) {
  cv::Mat img = cv::imread(img_path);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  return img;
}
