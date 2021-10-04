#include <iostream>
#include <torch/script.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

cv::Mat readImg(std::string img_path) {
  cv::Mat img = cv::imread(img_path);
  cvtColor(img, img, cv::COLOR_BGR2RGB);
  return img;
}

cv::Mat resizeNormalizeImg(cv::Mat img) {
  int img_h = 224, img_w = 224;
  if(img.rows == img_h && img.cols == img_w)
    return img;
  cv::Mat img_resized = cv::Mat(img_h, img_w, CV_8UC3);
  resize(img, img_resized, img_resized.size(), cv::INTER_AREA);
  img_resized.convertTo(img_resized, CV_32F);
  img_resized -= cv::Scalar(70.370258669401, 74.122797553143, 73.178495195211);   // Subtract mean
  img_resized /= cv::Scalar(51.644749102296, 51.515240911425, 52.277554279530);   // Divide Std
  return img_resized;
}

torch::Tensor getInputTensor(cv::Mat img) {
  torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, img.channels()}, at::kByte);
  tensor = tensor.permute(torch::IntList({0, 3, 1, 2}));    // Channel first ordering for torch
  tensor = tensor.toType(torch::kFloat32);                  // Covert uint8 to float32
  tensor = tensor.contiguous();                             // Make contiguous for torch.view()
  return tensor;
}

void printMatDetails(cv::Mat mat, std::string desc="Matrix") {
  std::string typeString;
  int type = mat.type();

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  typeString = "8U"; break;
    case CV_8S:  typeString = "8S"; break;
    case CV_16U: typeString = "16U"; break;
    case CV_16S: typeString = "16S"; break;
    case CV_32S: typeString = "32S"; break;
    case CV_32F: typeString = "32F"; break;
    case CV_64F: typeString = "64F"; break;
    default:     typeString = "User"; break;
  }

  typeString += "C";
  typeString += (chans+'0');

  double min, max;
  cv::minMaxLoc(mat, &min, &max);
  if(min < 0.6 || max > 1)
    printf("%s: %s %dx%d [%.2f, %.2f]\n", desc.c_str(), typeString.c_str(), mat.cols, mat.rows, min, max);

}

int main(int argc, char **argv) {

    if (argc < 3) {
    std::cout << "Usage: " << argv[0] << "<pth_model_path> <img_dir>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string& model_path = argv[1];
  const std::string& imgs_dir = argv[2];
  bool usegpu = true;

  // Build model
  torch::Device device(usegpu ? torch::kCUDA : torch::kCPU);
  torch::Device cpu(torch::kCPU);
  torch::jit::script::Module model = torch::jit::load(model_path);
  model.to(device);
  model.eval();

  for (const auto & entry : fs::directory_iterator(imgs_dir)) {
    if(entry.path().extension() != ".png") continue;
    std::string img_path = entry.path();

    // Resize image if necessary
    cv::Mat img = readImg(img_path);
    img = resizeNormalizeImg(img);
    printMatDetails(img, "Input image " + img_path);
    torch::Tensor inputTensor = getInputTensor(img);
    std::cout << "input shape: " << inputTensor.sizes() << std::endl;

    // Inference
    torch::NoGradGuard no_grad;
    torch::Tensor output = model.forward({inputTensor.to(device)}).toTensor().detach().to(cpu);    // If model has 1 output
    cv::Mat descriptor = cv::Mat(1, 2048, CV_32F, output.data_ptr());
    descriptor.convertTo(descriptor, CV_64F);

    std::cout << "output shape: " << output.sizes() << std::endl;
    printMatDetails(descriptor, "Descriptor");
    std::cout << std::endl;
  }

  return 0;
}
