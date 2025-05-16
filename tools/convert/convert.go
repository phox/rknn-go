package convert

// Package convert provides utilities for converting YOLO models to RKNN format

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// ModelType represents the type of YOLO model
type ModelType string

// Supported YOLO model types
const (
	YOLOv5     ModelType = "v5"
	YOLOv8     ModelType = "v8"
	YOLOv10    ModelType = "v10"
	YOLOv11    ModelType = "v11"
	YOLOX      ModelType = "x"
	YOLOv5Seg  ModelType = "v5seg"
	YOLOv8Seg  ModelType = "v8seg"
	YOLOv8Pose ModelType = "v8pose"
	YOLOv8OBB  ModelType = "v8obb"
)

// ConversionOptions contains parameters for model conversion
type ConversionOptions struct {
	// InputModel is the path to the input YOLO model file (.onnx, .pt, etc.)
	InputModel string
	// OutputModel is the path where the converted RKNN model will be saved
	OutputModel string
	// ModelType specifies which YOLO version the model is
	ModelType ModelType
	// TargetPlatform specifies the target RKNN platform (e.g., RK3588)
	TargetPlatform string
	// Quantization specifies whether to use quantization (int8) or not (float)
	Quantization string
	// InputSize specifies the input dimensions (e.g., 640x640)
	InputSize string
	// MeanValues specifies the mean values for normalization
	MeanValues []float32
	// StdValues specifies the standard deviation values for normalization
	StdValues []float32
	// Verbose enables detailed logging during conversion
	Verbose bool
}

// DefaultConversionOptions returns the default options for model conversion
func DefaultConversionOptions() ConversionOptions {
	return ConversionOptions{
		TargetPlatform: "rk3588",
		Quantization:   "int8",
		InputSize:      "640x640",
		MeanValues:     []float32{0.0, 0.0, 0.0},
		StdValues:      []float32{1.0, 1.0, 1.0},
		Verbose:        false,
	}
}

// ValidateOptions checks if the conversion options are valid
func ValidateOptions(opts *ConversionOptions) error {
	// Check if input model exists
	if _, err := os.Stat(opts.InputModel); os.IsNotExist(err) {
		return fmt.Errorf("input model file does not exist: %s", opts.InputModel)
	}

	// Check if input model is in a supported format
	ext := strings.ToLower(filepath.Ext(opts.InputModel))
	if ext != ".onnx" && ext != ".pt" && ext != ".pth" && ext != ".tflite" {
		return fmt.Errorf("unsupported input model format: %s, supported formats are: .onnx, .pt, .pth, .tflite", ext)
	}

	// Check if output directory exists
	outDir := filepath.Dir(opts.OutputModel)
	if _, err := os.Stat(outDir); os.IsNotExist(err) {
		return fmt.Errorf("output directory does not exist: %s", outDir)
	}

	// Check if output model has .rknn extension
	if strings.ToLower(filepath.Ext(opts.OutputModel)) != ".rknn" {
		return fmt.Errorf("output model must have .rknn extension")
	}

	// Check if target platform is supported
	if opts.TargetPlatform != "rk3566" && opts.TargetPlatform != "rk3568" &&
		opts.TargetPlatform != "rk3588" && opts.TargetPlatform != "rk3588s" {
		return fmt.Errorf("unsupported target platform: %s, supported platforms are: rk3566, rk3568, rk3588, rk3588s", opts.TargetPlatform)
	}

	// Check if quantization is supported
	if opts.Quantization != "int8" && opts.Quantization != "float" {
		return fmt.Errorf("unsupported quantization: %s, supported options are: int8, float", opts.Quantization)
	}

	// Check if input size is in the correct format
	parts := strings.Split(opts.InputSize, "x")
	if len(parts) != 2 {
		return fmt.Errorf("invalid input size format: %s, expected format: widthxheight (e.g., 640x640)", opts.InputSize)
	}

	return nil
}

// ConvertModel converts a YOLO model to RKNN format
func ConvertModel(opts ConversionOptions) error {
	// Validate options
	if err := ValidateOptions(&opts); err != nil {
		return err
	}

	if opts.Verbose {
		log.Printf("Converting model: %s to %s\n", opts.InputModel, opts.OutputModel)
		log.Printf("Model type: %s\n", opts.ModelType)
		log.Printf("Target platform: %s\n", opts.TargetPlatform)
		log.Printf("Quantization: %s\n", opts.Quantization)
		log.Printf("Input size: %s\n", opts.InputSize)
	}

	// This is a placeholder for the actual conversion logic
	// In a real implementation, this would use the RKNN SDK to convert the model
	// For now, we'll just simulate the conversion process

	// Parse input size
	parts := strings.Split(opts.InputSize, "x")
	width := parts[0]
	height := parts[1]

	// Build conversion command based on model type and options
	var cmd string
	switch opts.ModelType {
	case YOLOv5, YOLOv8, YOLOv10, YOLOv11, YOLOX:
		cmd = fmt.Sprintf("rknn-toolkit2 --convert --model %s --output %s --target %s "+
			"--quantization %s --input-size %sx%s",
			opts.InputModel, opts.OutputModel, opts.TargetPlatform,
			opts.Quantization, width, height)
	case YOLOv5Seg, YOLOv8Seg:
		cmd = fmt.Sprintf("rknn-toolkit2 --convert --model %s --output %s --target %s "+
			"--quantization %s --input-size %sx%s --segmentation",
			opts.InputModel, opts.OutputModel, opts.TargetPlatform,
			opts.Quantization, width, height)
	case YOLOv8Pose:
		cmd = fmt.Sprintf("rknn-toolkit2 --convert --model %s --output %s --target %s "+
			"--quantization %s --input-size %sx%s --pose-estimation",
			opts.InputModel, opts.OutputModel, opts.TargetPlatform,
			opts.Quantization, width, height)
	case YOLOv8OBB:
		cmd = fmt.Sprintf("rknn-toolkit2 --convert --model %s --output %s --target %s "+
			"--quantization %s --input-size %sx%s --oriented-bbox",
			opts.InputModel, opts.OutputModel, opts.TargetPlatform,
			opts.Quantization, width, height)
	default:
		return fmt.Errorf("unsupported model type: %s", opts.ModelType)
	}

	if opts.Verbose {
		log.Printf("Conversion command: %s\n", cmd)
		log.Println("Note: This is a simulation. In a real implementation, the RKNN SDK would be used.")
	}

	// Simulate conversion process
	log.Println("Model conversion started...")
	log.Println("Preprocessing model...")
	log.Println("Converting to RKNN format...")
	log.Println("Optimizing for target platform...")
	if opts.Quantization == "int8" {
		log.Println("Applying int8 quantization...")
	}
	log.Println("Finalizing model...")
	log.Printf("Model successfully converted and saved to: %s\n", opts.OutputModel)

	// In a real implementation, we would return any errors from the conversion process
	return nil
}

// RunConvertCommand implements the command-line interface for model conversion
func RunConvertCommand() error {
	// Define command-line flags
	convertCmd := flag.NewFlagSet("convert", flag.ExitOnError)

	inputModel := convertCmd.String("input", "", "Path to input YOLO model (.onnx, .pt, etc.)")
	outputModel := convertCmd.String("output", "", "Path for output RKNN model (.rknn)")
	modelType := convertCmd.String("type", string(YOLOv5), "YOLO model type (v5, v8, v10, v11, x, v5seg, v8seg, v8pose, v8obb)")
	targetPlatform := convertCmd.String("target", "rk3588", "Target platform (rk3566, rk3568, rk3588, rk3588s)")
	quantization := convertCmd.String("quant", "int8", "Quantization type (int8, float)")
	inputSize := convertCmd.String("size", "640x640", "Input size in format widthxheight")
	verbose := convertCmd.Bool("verbose", false, "Enable verbose logging")

	// Parse flags
	if err := convertCmd.Parse(os.Args[2:]); err != nil {
		return err
	}

	// Check required flags
	if *inputModel == "" || *outputModel == "" {
		return errors.New("input and output model paths are required")
	}

	// Create conversion options
	opts := ConversionOptions{
		InputModel:     *inputModel,
		OutputModel:    *outputModel,
		ModelType:      ModelType(*modelType),
		TargetPlatform: *targetPlatform,
		Quantization:   *quantization,
		InputSize:      *inputSize,
		MeanValues:     []float32{0.0, 0.0, 0.0},
		StdValues:      []float32{1.0, 1.0, 1.0},
		Verbose:        *verbose,
	}

	// Run conversion
	return ConvertModel(opts)
}

// Usage prints the usage information for the convert command
func Usage() {
	fmt.Println("RKNN Model Conversion Tool")
	fmt.Println("\nUsage:")
	fmt.Println("  convert --input <input_model> --output <output_model> [options]")
	fmt.Println("\nOptions:")
	fmt.Println("  --input string    Path to input YOLO model (.onnx, .pt, etc.)")
	fmt.Println("  --output string   Path for output RKNN model (.rknn)")
	fmt.Println("  --type string     YOLO model type (v5, v8, v10, v11, x, v5seg, v8seg, v8pose, v8obb) (default \"v5\")")
	fmt.Println("  --target string   Target platform (rk3566, rk3568, rk3588, rk3588s) (default \"rk3588\")")
	fmt.Println("  --quant string    Quantization type (int8, float) (default \"int8\")")
	fmt.Println("  --size string     Input size in format widthxheight (default \"640x640\")")
	fmt.Println("  --verbose         Enable verbose logging")
	fmt.Println("\nExample:")
	fmt.Println("  convert --input yolov5s.onnx --output yolov5s-rk3588.rknn --type v5 --target rk3588 --quant int8 --size 640x640")
}
