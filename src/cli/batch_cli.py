
# Version 4.0 09/10/25
# Authors: Stephen Marsland, Nirosha Priyadarshani, Julius Juodakis, Virginia Listanti, Giotto Frean

#    AviaNZ bioacoustic analysis program
#    Copyright (C) 2017--2024

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# BatchCLI.py
#
# For running the batch processor from the terminal

from core.batch_processor import BatchProcessor, BatchProcessorCallbacks
from src.utils.exceptions import GentleExitException


class CLIUserInteraction(BatchProcessorCallbacks):
    """CLI implementation of user interaction callbacks"""
    
    def ask_resume_analysis(self, message):
        response = input(message + " [y/n]: ")
        return response.lower() in ['yes', 'y']
        
    def confirm_analysis_launch(self, message):
        response = input(message + " [y/n]: ")
        return response.lower() in ['yes', 'y']
        
    def update_progress(self, current, total, message):
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