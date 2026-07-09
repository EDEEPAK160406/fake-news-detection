"""
Quick start script - runs the entire OCR system pipeline
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import Logger


def print_header():
    """Print welcome header."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║     Image-to-Text Extraction Module for Fake News Detection      ║
║                         Quick Start Guide                         ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """Print main menu."""
    print("""
Select an option:

1. View Examples and Demonstrations
2. Generate Synthetic Training Data
3. Train a New Model
4. Run Inference on Image(s)
5. Run Unit Tests
6. View Documentation
7. Exit

    """)


def option_1_examples():
    """Run examples."""
    print("\nRunning examples...")
    from examples import run_all_examples
    run_all_examples()


def option_2_synthetic_data():
    """Generate synthetic data."""
    print("\nGenerating synthetic training data...")
    from src.train import SyntheticDataGenerator
    from src.utils import Logger
    
    logger = Logger.setup_logger('synthetic_data', 'INFO')
    
    try:
        num_samples = input("Number of samples to generate (default 500): ")
        num_samples = int(num_samples) if num_samples else 500
        
        images, texts = SyntheticDataGenerator.generate_synthetic_dataset(
            num_samples=num_samples
        )
        
        logger.info(f"\nDataset generated successfully!")
        logger.info(f"Images shape: {images.shape}")
        logger.info(f"Number of text samples: {len(texts)}")
        logger.info(f"Sample texts: {texts[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")


def option_3_train_model():
    """Train model."""
    print("\nModel Training Interface")
    from src.train import OCRTrainer, SyntheticDataGenerator
    from src.utils import Logger
    
    logger = Logger.setup_logger('training', 'INFO')
    
    try:
        # Generate data
        print("Generating training data...")
        num_samples = input("Number of training samples (default 1000): ")
        num_samples = int(num_samples) if num_samples else 1000
        
        images, texts = SyntheticDataGenerator.generate_synthetic_dataset(
            num_samples=num_samples
        )
        
        labels = SyntheticDataGenerator.text_to_labels(texts)
        
        # Train
        print("Initializing trainer...")
        trainer = OCRTrainer()
        trainer.build_model()
        
        print("Starting training (this will take a while)...")
        trainer.train(images, labels, augment=True)
        
        # Save
        model_path = input("Model save path (default: models/custom_model.h5): ")
        model_path = model_path if model_path else "models/custom_model.h5"
        
        trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        print(f"Error: {e}")


def option_4_inference():
    """Run inference."""
    print("\nInference Interface")
    from src.ocr_extractor import create_ocr_extractor
    from src.utils import Logger
    
    logger = Logger.setup_logger('inference', 'INFO')
    
    try:
        mode = input("Select mode (1=single image, 2=batch): ").strip()
        
        model_path = input("Model path (default: models/best_ocr_model.h5): ")
        model_path = model_path if model_path else "models/best_ocr_model.h5"
        
        use_fake_news = input("Use fake news detection mode? (y/n): ").lower() == 'y'
        
        # Create extractor
        extractor = create_ocr_extractor(
            model_path=model_path,
            for_fake_news=use_fake_news
        )
        
        if mode == '1':
            # Single image
            image_path = input("Image path: ")
            
            if use_fake_news:
                result = extractor.extract_for_analysis(image_path)
            else:
                result = extractor.extract_text(image_path)
            
            print(f"\nExtracted Text: {result['extracted_text']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
        elif mode == '2':
            # Batch
            batch_dir = input("Batch directory path: ")
            output_file = input("Output file (default: results.json): ")
            output_file = output_file if output_file else "results.json"
            
            from pathlib import Path
            image_paths = list(Path(batch_dir).glob('*.jpg')) + list(Path(batch_dir).glob('*.png'))
            
            results = extractor.extract_batch([str(p) for p in image_paths])
            
            from src.utils import ResultsFormatter
            ResultsFormatter.save_results_json(
                {'total_images': len(results), 'results': results},
                output_file
            )
            
            print(f"Processed {len(results)} images")
            print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")


def option_5_tests():
    """Run tests."""
    print("\nRunning unit tests...")
    import unittest
    
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def option_6_documentation():
    """Show documentation."""
    print("\n" + "="*70)
    print("DOCUMENTATION")
    print("="*70)
    
    readme_path = Path(__file__).parent / 'README.md'
    
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()
            # Print first 2000 chars
            print(content[:2000])
            print("\n... (see README.md for full documentation)")
    else:
        print("README.md not found")


def main():
    """Main quick start interface."""
    print_header()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            option_1_examples()
        elif choice == '2':
            option_2_synthetic_data()
        elif choice == '3':
            option_3_train_model()
        elif choice == '4':
            option_4_inference()
        elif choice == '5':
            option_5_tests()
        elif choice == '6':
            option_6_documentation()
        elif choice == '7':
            print("\nThank you for using OCR System. Goodbye! 👋")
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
        print("\n" * 2)


if __name__ == '__main__':
    main()
