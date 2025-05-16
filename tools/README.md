# RKNN Model Conversion Tools

This directory contains tools for working with RKNN models, specifically for converting YOLO models to the RKNN format for use with Rockchip NPU hardware.

## Model Converter

The model converter tool allows you to convert YOLO models (in ONNX, PyTorch, or TFLite format) to RKNN format for deployment on Rockchip NPU devices.

### Supported YOLO Models

- YOLOv5
- YOLOv8
- YOLOv10
- YOLOv11
- YOLOX
- YOLOv5-Segmentation
- YOLOv8-Segmentation
- YOLOv8-Pose
- YOLOv8-OBB (Oriented Bounding Box)

### Supported Target Platforms

- RK3566
- RK3568
- RK3588
- RK3588S

### Usage

```bash
go run main.go convert --input <input_model> --output <output_model> [options]
```

### Options

- `--input string`: Path to input YOLO model (.onnx, .pt, .pth, .tflite)
- `--output string`: Path for output RKNN model (.rknn)
- `--type string`: YOLO model type (v5, v8, v10, v11, x, v5seg, v8seg, v8pose, v8obb) (default "v5")
- `--target string`: Target platform (rk3566, rk3568, rk3588, rk3588s) (default "rk3588")
- `--quant string`: Quantization type (int8, float) (default "int8")
- `--size string`: Input size in format widthxheight (default "640x640")
- `--verbose`: Enable verbose logging

### Example

```bash
go run main.go convert --input yolov5s.onnx --output yolov5s-rk3588.rknn --type v5 --target rk3588 --quant int8 --size 640x640
```

## Building the Tool

To build the tool as a standalone executable:

```bash
cd tools
go build -o rknn-convert
```

Then you can use it directly:

```bash
./rknn-convert convert --input yolov5s.onnx --output yolov5s-rk3588.rknn
```

## Notes

- This tool currently provides a simulation of the conversion process. In a real implementation, it would use the RKNN SDK to perform the actual conversion.
- The conversion process requires the RKNN Toolkit to be installed on your system.
- For optimal performance, it's recommended to use the int8 quantization option, which provides faster inference with minimal accuracy loss.
- Make sure to specify the correct input size that matches your model's expected input dimensions.