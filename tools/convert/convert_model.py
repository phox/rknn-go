#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RKNN Model Converter Script

This script serves as a bridge between Go code and the RKNN Toolkit Python package.
It takes command line arguments and uses them to convert models to RKNN format.
"""

import argparse
import os
import sys
import logging

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
    parser.add_argument('--segmentation', action='store_true', help='Enable segmentation model conversion')
    parser.add_argument('--pose-estimation', action='store_true', help='Enable pose estimation model conversion')
    parser.add_argument('--oriented-bbox', action='store_true', help='Enable oriented bounding box model conversion')
    parser.add_argument('--mean', default='0.0,0.0,0.0', help='Mean values for normalization')
    parser.add_argument('--std', default='1.0,1.0,1.0', help='Standard deviation values for normalization')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()

def convert_model(args):
    """Convert model to RKNN format using RKNN Toolkit."""
    try:
        # Import RKNN Toolkit here to avoid import errors if not installed
        from rknn.api import RKNN
    except ImportError:
        logger.error("Failed to import RKNN Toolkit. Please make sure it's installed.")
        return 1
    
    # Parse input size
    width, height = args.input_size.split('x')
    # Create input size as a list of dimensions
    input_size = [1, 3, int(height), int(width)]  # [batch, channels, height, width]
    
    # Parse mean and std values
    mean_values = [float(x) for x in args.mean.split(',')]
    std_values = [float(x) for x in args.std.split(',')]
    
    if args.verbose:
        logger.info(f"Converting model: {args.model} to {args.output}")
        logger.info(f"Target platform: {args.target}")
        logger.info(f"Quantization: {args.quantization}")
        logger.info(f"Input size: {args.input_size}")
    
    # Initialize RKNN object
    rknn = RKNN(verbose=args.verbose)
    
    # Pre-process config
    print('--> Config model')
    config_params = {
        'mean_values': mean_values,
        'std_values': std_values,
        'target_platform': args.target
    }
    
    # Only set quantized_algorithm for int8 quantization
    if args.quantization == 'int8':
        config_params['quantized_algorithm'] = 'normal'
        
    # Add dynamic_input parameter to handle dynamic input shapes
    # dynamic_input needs to be a list of lists for RKNN Toolkit 2.3.2+
    # Format should be [input_size] where input_size is [batch, channels, height, width]
    # The error message indicates that dynamic_input[0] should be a single element, not a pair
    config_params['dynamic_input'] = [input_size]
        
    rknn.config(**config_params)
    print('done')
    
    # Load model
    print('--> Loading model')
    if args.model.lower().endswith('.onnx'):
        # For ONNX models, specify input_size_list to fix input shape issues
        # Use the same format as dynamic_input for consistency
        ret = rknn.load_onnx(model=args.model, input_size_list=[input_size])
    else:
        ret = rknn.load_pytorch(model=args.model)
    if ret != 0:
        logger.error(f"Failed to load model: {args.model}")
        return ret
    print('done')
    
    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=args.quantization == 'int8')
    if ret != 0:
        logger.error("Failed to build model")
        return ret
    print('done')
    
    # Export RKNN model
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        logger.error(f"Failed to export RKNN model: {args.output}")
        return ret
    print('done')
    
    # Verify the output file was created
    if not os.path.exists(args.output):
        logger.error(f"Conversion completed but output file was not created: {args.output}")
        return 1
    
    logger.info(f"Model successfully converted and saved to: {args.output}")
    return 0

def main():
    """Main function."""
    args = parse_args()
    return convert_model(args)

if __name__ == '__main__':
    sys.exit(main())