# Libtorch CV Infer

Inference with libtorch and opencv in C++

## Dependencies

- OpenCV (tested with 4.5.3)
- Libtorch (tested with 1.9.0 cu111)

## Quick Start

```
git clone https://github.com/saravanabalagi/libtorch_cv_infer.git
sh build.sh
```

Note that this will run [setup.sh](setup.sh) which will download OpenCV and libtorch. Once it's built successfully, you can run it using

```
cd build
./predict model.pth img.png
./predict_dir model.pth imgs/
```

Currently the predict only shows info and does not save the embedding vector to disk. Also, the image preprocessing is setup for 224 x 224 x 3 resolution input ResNet pre-trained on ImageNet. You will need to configure the image preprocessing pipeline accordingly as you use different models.
