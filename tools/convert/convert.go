package convert

// Package convert provides utilities for converting YOLO models to RKNN format

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
	// DynamicInput specifies whether to enable dynamic input shape (advanced, only if needed)
	DynamicInput bool
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
		DynamicInput:   false,
	}
}

// isDefaultMeanStd checks if the provided mean and std values are the default ones
func isDefaultMeanStd(mean, std []float32) bool {
	defaultMean := []float32{0.0, 0.0, 0.0}
	defaultStd := []float32{1.0, 1.0, 1.0}

	if len(mean) != len(defaultMean) || len(std) != len(defaultStd) {
		return false
	}

	for i := 0; i < len(mean); i++ {
		if mean[i] != defaultMean[i] {
			return false
		}
	}

	for i := 0; i < len(std); i++ {
		if std[i] != defaultStd[i] {
			return false
		}
	}

	return true
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

// ConvertModel converts a YOLO model to RKNN format using the RKNN SDK
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

	// Parse input size
	parts := strings.Split(opts.InputSize, "x")
	width := parts[0]
	height := parts[1]

	// Check for unsupported model type
	switch opts.ModelType {
	case YOLOv5, YOLOv8, YOLOv10, YOLOv11, YOLOX, YOLOv5Seg, YOLOv8Seg, YOLOv8Pose, YOLOv8OBB:
		// Supported model types
	default:
		return fmt.Errorf("unsupported model type: %s", opts.ModelType)
	}

	// Prepare arguments for the Python script
	// Get the directory of the current Go file for more reliable script location
	_, currentFilePath, _, _ := runtime.Caller(0)
	scriptPath := filepath.Join(filepath.Dir(currentFilePath), "convert_model.py")

	// Build Python script arguments
	pyArgs := []string{
		scriptPath,
		"--model", opts.InputModel,
		"--output", opts.OutputModel,
		"--target", opts.TargetPlatform,
		"--quantization", opts.Quantization,
		"--input-size", fmt.Sprintf("%sx%s", width, height),
	}

	// Add model-specific arguments
	switch opts.ModelType {
	case YOLOv5Seg, YOLOv8Seg:
		pyArgs = append(pyArgs, "--segmentation")
	case YOLOv8Pose:
		pyArgs = append(pyArgs, "--pose-estimation")
	case YOLOv8OBB:
		pyArgs = append(pyArgs, "--oriented-bbox")
	}

	// Add mean and std values if they're not the default
	if !isDefaultMeanStd(opts.MeanValues, opts.StdValues) {
		meanStr := fmt.Sprintf("%f,%f,%f", opts.MeanValues[0], opts.MeanValues[1], opts.MeanValues[2])
		stdStr := fmt.Sprintf("%f,%f,%f", opts.StdValues[0], opts.StdValues[1], opts.StdValues[2])
		pyArgs = append(pyArgs, "--mean", meanStr, "--std", stdStr)
	}

	if opts.Verbose {
		pyArgs = append(pyArgs, "--verbose")
	}
	if opts.DynamicInput {
		pyArgs = append(pyArgs, "--dynamic-input")
	}

	log.Printf("Running RKNN conversion with Python script: python %s\n", strings.Join(pyArgs, " "))

	// Execute the Python script
	cmd := exec.Command("python", pyArgs...)

	// Create pipes for stdout and stderr
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start conversion command: %v", err)
	}

	// Create a channel to signal when processing is complete
	done := make(chan bool)

	// Process stdout in a goroutine
	go func() {
		scanner := bufio.NewScanner(stdoutPipe)
		for scanner.Scan() {
			line := scanner.Text()
			log.Println(line)
		}
		done <- true
	}()

	// Process stderr in a goroutine
	go func() {
		scanner := bufio.NewScanner(stderrPipe)
		for scanner.Scan() {
			line := scanner.Text()
			log.Println("ERROR:", line)
		}
		done <- true
	}()

	// Wait for both stdout and stderr processing to complete
	<-done
	<-done

	// Wait for the command to complete
	err = cmd.Wait()
	if err != nil {
		return fmt.Errorf("model conversion failed: %v", err)
	}

	// Verify the output file was created
	if _, err := os.Stat(opts.OutputModel); os.IsNotExist(err) {
		return fmt.Errorf("conversion completed but output file was not created: %s", opts.OutputModel)
	}

	log.Printf("Model successfully converted and saved to: %s\n", opts.OutputModel)
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
	dynamicInput := convertCmd.Bool("dynamic-input", false, "Enable dynamic input shape (advanced, only if needed)")

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
		DynamicInput:   *dynamicInput,
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
	fmt.Println("  --dynamic-input   Enable dynamic input shape (advanced, only if needed)")
	fmt.Println("\nExample:")
	fmt.Println("  convert --input yolov5s.onnx --output yolov5s-rk3588.rknn --type v5 --target rk3588 --quant int8 --size 640x640")
}
