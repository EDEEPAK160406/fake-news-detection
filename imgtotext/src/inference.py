"""
Inference/prediction script for OCR model.
"""

import argparse
import json
from pathlib import Path
import logging
from typing import List

from src.ocr_extractor import create_ocr_extractor, OCRExtractor, FakeNewsTextExtractor
from src.utils import Logger, ResultsFormatter, ImageLoader
from config.config import *


def main():
    """Main inference script."""
    parser = argparse.ArgumentParser(description='OCR Text Extraction Inference')
    
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--batch', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, default=f'{MODEL_DIR}/{BEST_MODEL_NAME}',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file path')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for processing')
    parser.add_argument('--confidence-threshold', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Minimum confidence threshold')
    parser.add_argument('--fake-news', action='store_true',
                       help='Use fake news detection mode')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = Logger.setup_logger('inference', LOG_LEVEL)
    
    logger.info("OCR Inference Pipeline")
    logger.info("=" * 50)
    
    # Create configuration
    config = {
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'characters': CHARACTERS,
        'num_classes': NUM_CLASSES,
        'rnn_units': RNN_UNITS,
        'confidence_threshold': args.confidence_threshold,
        'beam_width': CTCBeamSearchDecoder_BEAM_WIDTH,
        'model_type': 'cnn_lstm_ctc'
    }
    
    # Initialize OCR extractor
    logger.info(f"Loading model from {args.model}")
    try:
        extractor = create_ocr_extractor(
            model_path=args.model,
            config=config,
            for_fake_news=args.fake_news
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Process single image
    if args.image:
        logger.info(f"Processing single image: {args.image}")
        
        if args.fake_news:
            result = extractor.extract_for_analysis(args.image)
        else:
            result = extractor.extract_text(args.image)
        
        results = {'total_images': 1, 'results': [result]}
        
        logger.info(f"Extracted text: {result['extracted_text']}")
        logger.info(f"Confidence: {result['confidence']:.4f}")
        
    # Process batch of images
    elif args.batch:
        logger.info(f"Processing batch from directory: {args.batch}")
        
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            logger.error(f"Directory not found: {args.batch}")
            return
        
        # Get all image files
        image_files = list(batch_dir.glob('*.jpg')) + list(batch_dir.glob('*.png'))
        image_files += list(batch_dir.glob('*.jpeg')) + list(batch_dir.glob('*.bmp'))
        
        if not image_files:
            logger.warning(f"No images found in {args.batch}")
            return
        
        logger.info(f"Found {len(image_files)} images")
        
        # Process batch
        image_paths = [str(f) for f in image_files]
        
        if args.fake_news:
            extracted_results = [
                extractor.extract_for_analysis(p) for p in image_paths
            ]
        else:
            extracted_results = extractor.extract_batch(
                image_paths,
                batch_size=args.batch_size
            )
        
        # Filter by confidence
        filtered_results = [
            r for r in extracted_results
            if r.get('confidence', 0) >= args.confidence_threshold
        ]
        
        logger.info(f"Successfully processed: {len(filtered_results)}/{len(extracted_results)}")
        
        results = {
            'total_images': len(image_paths),
            'successfully_processed': len(filtered_results),
            'average_confidence': sum(r.get('confidence', 0) for r in filtered_results) / len(filtered_results) if filtered_results else 0,
            'results': filtered_results
        }
    
    else:
        parser.print_help()
        return
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == 'json':
        ResultsFormatter.save_results_json(results, str(output_path))
    elif args.format == 'csv':
        ResultsFormatter.save_results_csv(results, str(output_path))
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    logger.info("=" * 50)
    logger.info(f"Total images processed: {results.get('total_images', 1)}")
    logger.info(f"Successfully processed: {results.get('successfully_processed', len(results.get('results', [])))}")
    logger.info(f"Average confidence: {results.get('average_confidence', 0):.4f}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
