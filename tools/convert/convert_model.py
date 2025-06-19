#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RKNN Model Converter Script

This script serves as a bridge between Go code and the RKNN Toolkit Python package.
It uses rknn_convert tool to convert models to RKNN format.
"""

import argparse
import os
import sys
import logging
import yaml
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert models to RKNN format')
    parser.add_argument('--model', required=True, help='Path to input model')
    parser.add_argument('--output', required=True, help='Path for output RKNN model')
    parser.add_argument('--target', default='rk3588', help='Target platform')
    parser.add_argument('--quantization', default='int8', help='Quantization type')
    parser.add_argument('--input-size', default='640x640', help='Input size in format widthxheight')
    parser.add_argument('--model-type', default='yolov5', choices=['yolov5', 'yolov8', 'yolov10'], help='YOLO model type')
    parser.add_argument('--segmentation', action='store_true', help='Enable segmentation model conversion')
    parser.add_argument('--pose-estimation', action='store_true', help='Enable pose estimation model conversion')
    parser.add_argument('--oriented-bbox', action='store_true', help='Enable oriented bounding box model conversion')
    parser.add_argument('--mean', default='0.0,0.0,0.0', help='Mean values for normalization')
    parser.add_argument('--std', default='1.0,1.0,1.0', help='Standard deviation values for normalization')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--dynamic-input', action='store_true', help='Enable dynamic input shape (advanced, only if needed)')
    
    return parser.parse_args()

def create_config_yml(args, temp_dir):
    """Create YAML configuration file for rknn_convert."""
    # Parse mean and std values
    mean_values = [float(x) for x in args.mean.split(',')]
    std_values = [float(x) for x in args.std.split(',')]
    
    # Parse input size
    width, height = map(int, args.input_size.split('x'))
    
    # Determine model type for custom_string
    model_type = args.model_type
    if args.segmentation:
        model_type += '-seg'
    if args.pose_estimation:
        model_type += '-pose'
    if args.oriented_bbox:
        model_type += '-obb'
    
    # Create config dictionary
    config = {
        'models': {
            'name': Path(args.output).stem,
            'platform': 'onnx' if args.model.lower().endswith('.onnx') else 'pytorch',
            'model_file_path': args.model,
            'quantize': args.quantization == 'int8',
            'configs': {
                'mean_values': mean_values,
                'std_values': std_values,
                'quant_img_RGB2BGR': False,
                'quantized_algorithm': 'normal',
                'quantized_method': 'channel',
                'quantized_dtype': 'w8a8',
                'float_dtype': 'float16',
                'optimization_level': 3,
                'custom_string': model_type,
                'inputs': ['images'],
                'input_size_list': [[1, 3, height, width]]
                
            }
        }
    }
    
    # Write config to temporary file
    config_path = os.path.join(temp_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path

def convert_model(args):
    """Convert model to RKNN format using rknn_convert."""
    try:
        # Create temporary directory for config file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create YAML configuration
            config_path = create_config_yml(args, temp_dir)
            logger.info(f"Created config file at: {config_path}")
            
            # Build rknn_convert command
            cmd = [
                'python', '-m', 'rknn.api.rknn_convert',
                '-t', args.target,
                '-i', config_path,
                '-o', args.output
            ]
            
            # Add optional arguments
            if args.verbose:
                cmd.append('-v')
            
            # Log the command
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Execute conversion
            print('--> Converting model')
            import subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Log the output
            if result.stdout:
                logger.info(f"Command stdout:\n{result.stdout}")
            if result.stderr:
                logger.error(f"Command stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                logger.error(f"Conversion failed with return code: {result.returncode}")
                return result.returncode
            
            print(result.stdout)
            print('done')
            
            # Verify the output file was created
            if not os.path.exists(args.output):
                logger.error(f"Conversion completed but output file was not created: {args.output}")
                return 1
            
            logger.info(f"Model successfully converted and saved to: {args.output}")
            return 0
            
    except Exception as e:
        logger.error(f"Conversion failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return 1

def main():
    """Main function."""
    args = parse_args()
    return convert_model(args)

if __name__ == '__main__':
    sys.exit(main())