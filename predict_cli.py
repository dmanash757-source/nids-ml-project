#!/usr/bin/env python3
"""
Standalone predictor CLI for the NIDS project.

Usage:
    python predict_cli.py                    # Interactive mode
    python predict_cli.py --sample normal     # Test normal sample
    python predict_cli.py --sample attack     # Test attack sample
    python predict_cli.py --file input.csv    # Predict from CSV file
"""

import os
import sys
import argparse
import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.predictor import IntrusionPredictor, DEMO_NORMAL_SAMPLE, DEMO_ATTACK_SAMPLE


MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "intrusion_model.pkl")
PREP_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")


SELECTED_FEATURES = [
    'flag', 'same_srv_rate', 'logged_in', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'srv_serror_rate', 'diff_srv_rate',
    'protocol_type', 'serror_rate', 'dst_host_same_src_port_rate',
    'dst_host_diff_srv_rate', 'service', 'dst_host_srv_diff_host_rate',
    'count', 'dst_host_rerror_rate', 'dst_host_count', 'dst_host_srv_rerror_rate',
    'dst_host_srv_serror_rate', 'srv_count', 'dst_host_serror_rate'
]


def main():
    parser = argparse.ArgumentParser(description="NIDS Predictor CLI")
    parser.add_argument("--sample", choices=["normal", "attack"], 
                       help="Test with built-in sample")
    parser.add_argument("--file", type=str, 
                       help="Predict from CSV file")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode")
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Run main.py first to train the model.")
        sys.exit(1)
    if not os.path.exists(PREP_PATH):
        print("Error: Preprocessor not found. Run main.py first.")
        sys.exit(1)
    
    # Load preprocessor
    preprocessor = joblib.load(PREP_PATH)
    
    # Create predictor with selected features
    predictor = IntrusionPredictor(
        model_path=MODEL_PATH, 
        preprocessor=preprocessor,
        selected_features=SELECTED_FEATURES
    )
    
    # Test with built-in sample
    if args.sample == "normal":
        print("Testing NORMAL sample...")
        result = predictor.predict_one(DEMO_NORMAL_SAMPLE)
        conf_str = f"{result['confidence']:.2%}" if result['confidence'] else "N/A"
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {conf_str}")
        
    elif args.sample == "attack":
        print("Testing ATTACK sample...")
        result = predictor.predict_one(DEMO_ATTACK_SAMPLE)
        conf_str = f"{result['confidence']:.2%}" if result['confidence'] else "N/A"
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {conf_str}")
    
    # Predict from CSV file
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        print(f"Predicting from file: {args.file}")
        df = pd.read_csv(args.file)
        
        # Predict each row
        predictions = []
        for idx, row in df.iterrows():
            sample = row.to_dict()
            result = predictor.predict_one(sample)
            predictions.append({
                "index": idx,
                "prediction": result['prediction'],
                "confidence": result['confidence']
            })
        
        results_df = pd.DataFrame(predictions)
        print(f"\nResults ({len(results_df)} samples):")
        print(results_df.to_string(index=False))
        
        # Save results
        output_path = args.file.replace(".csv", "_predictions.csv")
        results_df.to_csv(output_path, index=False)
        print(f"\n[+] Predictions saved to: {output_path}")
    
    # Interactive mode
    elif args.interactive or (not args.sample and not args.file):
        print("\n=== NIDS Interactive Predictor ===")
        print("Enter network traffic features (or 'quit' to exit)\n")
        
        while True:
            try:
                sample_type = input("Sample type (normal/attack): ").strip().lower()
                if sample_type == "quit" or sample_type == "q":
                    break
                elif sample_type == "normal":
                    result = predictor.predict_one(DEMO_NORMAL_SAMPLE)
                    conf_str = f"{result['confidence']:.2%}" if result['confidence'] else "N/A"
                    print(f"  → Prediction: {result['prediction']} (Confidence: {conf_str})")
                elif sample_type == "attack":
                    result = predictor.predict_one(DEMO_ATTACK_SAMPLE)
                    conf_str = f"{result['confidence']:.2%}" if result['confidence'] else "N/A"
                    print(f"  → Prediction: {result['prediction']} (Confidence: {conf_str})")
                else:
                    print("  Invalid choice. Use: normal, attack, or quit")
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
