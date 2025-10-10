#!/usr/bin/env python3

# cli_batch.py
# CLI interface for AviaNZ batch processing using the clean batch processor

import os
import sys
from src.core.batch_processor import BatchProcessor, BatchProcessorCallbacks
from src.utils.exceptions import GentleExitException


class CLIUserInteraction(BatchProcessorCallbacks):
    """CLI implementation of user interaction callbacks"""
    
    def ask_resume_analysis(self, message: str) -> bool:
        response = input(message + " [y/n]: ")
        return response.lower() in ['yes', 'y']
        
    def confirm_analysis_launch(self, message: str) -> bool:
        response = input(message + " [y/n]: ")
        return response.lower() in ['yes', 'y']
        
    def update_progress(self, current: int, total: int, message: str) -> None:
        print(f"Progress: {current}/{total} - {message}")

def run_cli_batch(configdir, directory, recognisers, **kwargs):
    """Run batch processing in CLI mode"""
    
    # Create CLI callbacks
    callbacks = CLIUserInteraction()
    
    # Create and run the processor
    processor = BatchProcessor(
        configdir=configdir,
        directory=directory, 
        recognisers=recognisers,
        callbacks=callbacks,
        **kwargs
    )
    
    try:
        result = processor.process_files()
        if result == 0:
            print("✓ Analysis completed successfully")
            return 0
        else:
            print("✗ Analysis failed")
            return 1
    except GentleExitException:
        print("Analysis cancelled by user")
        return 1
    except Exception as e:
        print(f"✗ Analysis failed with error: {e}")
        return 1

if __name__ == "__main__":
    # Example usage - this would be called from AviaNZ.py
    import argparse
    
    parser = argparse.ArgumentParser(description="AviaNZ Batch Processing CLI")
    parser.add_argument("--configdir", required=True, help="Configuration directory")
    parser.add_argument("--directory", required=True, help="Directory to process")
    parser.add_argument("--recogniser", required=True, help="Recogniser to use")
    parser.add_argument("--subset", action="store_true", help="Use time subset")
    parser.add_argument("--intermittent", action="store_true", help="Use intermittent sampling")
    parser.add_argument("--wind", default="None", help="Wind filtering")
    parser.add_argument("--merge-syllables", action="store_true", help="Merge syllables")
    parser.add_argument("--time-start", type=int, default=0, help="Start time (seconds from midnight)")
    parser.add_argument("--time-end", type=int, default=0, help="End time (seconds from midnight)")
    parser.add_argument("--protocol-size", type=int, default=15, help="Protocol size for intermittent sampling")
    parser.add_argument("--protocol-interval", type=int, default=300, help="Protocol interval for intermittent sampling")
    parser.add_argument("--maxgap", type=float, default=1.0, help="Maximum gap for syllable merging")
    parser.add_argument("--minlen", type=float, default=0.2, help="Minimum syllable length")
    parser.add_argument("--maxlen", type=float, default=10.0, help="Maximum syllable length")
    
    args = parser.parse_args()
    
    result = run_cli_batch(
        configdir=args.configdir,
        directory=args.directory,
        recognisers=[args.recogniser],
        subset=args.subset,
        intermittent=args.intermittent,
        wind=args.wind,
        mergeSyllables=args.merge_syllables,
        timeWindow_s=args.time_start,
        timeWindow_e=args.time_end,
        protocolSize=args.protocol_size,
        protocolInterval=args.protocol_interval,
        maxgap=args.maxgap,
        minlen=args.minlen,
        maxlen=args.maxlen
    )
    
    sys.exit(result)