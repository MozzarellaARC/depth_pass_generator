import torch
import cv2
import numpy as np
import argparse
import os
from pathlib import Path

class MiDaSDepthEstimator:
    """
    MiDaS Depth Estimation with multiple model options
    """
    
    AVAILABLE_MODELS = {
        "small": {
            "model_type": "MiDaS_small",
            "transform": "small_transform",
            "description": "Fastest, smallest model (~60MB) - good for quick processing",
            "quality": "Basic"
        },
        "hybrid": {
            "model_type": "DPT_Hybrid", 
            "transform": "dpt_transform",
            "description": "Medium size, much better quality than small",
            "quality": "Good"
        },
        "large": {
            "model_type": "DPT_Large",
            "transform": "dpt_transform", 
            "description": "Largest model, highest quality depth estimation",
            "quality": "Best"
        }
    }
    
    def __init__(self, model_name="hybrid"):
        """
        Initialize the depth estimator
        
        Args:
            model_name (str): One of 'small', 'hybrid', 'large'
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' not available. Choose from: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = self.AVAILABLE_MODELS[model_name]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading {model_name} model: {self.model_config['description']}")
        print(f"Using device: {self.device}")
        
        # Load model and transforms
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_config["model_type"])
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        
        # Get the appropriate transform
        self.transform = getattr(transforms, self.model_config["transform"])
        
        # Move model to device and set to eval mode
        self.model.to(self.device).eval()
        
        print(f"Model loaded successfully!")
    
    def estimate_depth(self, image_path, output_path=None):
        """
        Estimate depth from an input image
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save depth map (optional)
        
        Returns:
            numpy.ndarray: Normalized depth map
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transform and move to device
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Predict depth
        print("Predicting depth...")
        with torch.no_grad():
            prediction = self.model(input_batch)
        
        # Resize to original image dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Convert to numpy and normalize
        depth = prediction.cpu().numpy()
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        
        # Save depth map if output path provided
        if output_path:
            cv2.imwrite(output_path, depth_normalized.astype(np.uint8))
            print(f"Depth map saved to: {output_path}")
        
        return depth_normalized

def list_available_models():
    """Print available models and their descriptions"""
    print("\nAvailable MiDaS models:")
    print("-" * 50)
    for name, config in MiDaSDepthEstimator.AVAILABLE_MODELS.items():
        print(f"{name:8} | {config['quality']:5} | {config['description']}")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="MiDaS Depth Estimation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python midas_depth.py input.jpg --model small
    python midas_depth.py input.jpg --model hybrid --output my_depth.png
    python midas_depth.py input.jpg --model large --output depth_maps/result.png
    python midas_depth.py --list-models
        """
    )
    
    parser.add_argument("input_image", nargs="?", help="Path to input image")
    parser.add_argument("--model", "-m", 
                      choices=list(MiDaSDepthEstimator.AVAILABLE_MODELS.keys()),
                      default="hybrid",
                      help="Model to use for depth estimation (default: hybrid)")
    parser.add_argument("--output", "-o", 
                      help="Output path for depth map (default: depth_output.png)")
    parser.add_argument("--list-models", action="store_true",
                      help="List available models and exit")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        list_available_models()
        return
    
    # Validate input
    if not args.input_image:
        print("Error: Input image path is required")
        parser.print_help()
        return
    
    # Set default output path
    if not args.output:
        input_path = Path(args.input_image)
        args.output = f"depth_{input_path.stem}.png"
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize estimator
        estimator = MiDaSDepthEstimator(args.model)
        
        # Estimate depth
        depth_map = estimator.estimate_depth(args.input_image, args.output)
        
        print(f"\nDepth estimation completed successfully!")
        print(f"Model used: {args.model} ({estimator.model_config['quality']} quality)")
        print(f"Input: {args.input_image}")
        print(f"Output: {args.output}")
        print(f"Depth map shape: {depth_map.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()