# Bat Export Migration Guide

## Changes Made

The bat export methods have been **removed from `BatchProcessor`** as they violated separation of concerns. BatchProcessor should focus on processing audio files, not exporting results.

### Removed Methods:
- `BatchProcessor.exportBatResults()`
- `BatchProcessor.exportBatSurvey()`
- `BatchProcessor.exportToDOCDB()`

### Removed Callback:
- `BatchProcessorCallbacks.get_bat_survey_info()`

## How to Export Bat Results Now

Use `BatExporter` directly instead of going through `BatchProcessor`:

### Import BatExporter

```python
from src.core.BatExporter import BatExporter
from src.core.BatDetector import BatDetector
```

### Create BatExporter Instance

```python
# After processing, create the exporter
config = processor.config  # Or load config directly
bat_detector = BatDetector()  # Or reuse existing instance
bat_exporter = BatExporter(config, bat_detector)
```

### Export Results

#### XML Format (for BatSearch database)
```python
bat_exporter.exportResults(
    dirName='/path/to/results',
    format='xml',
    threshold1=0.85,
    threshold2=0.7
)
```

#### CSV Format
```python
bat_exporter.exportResults(
    dirName='/path/to/results',
    format='csv',
    threshold1=0.85,
    threshold2=0.7
)
```

#### Passes Format
```python
bat_exporter.exportResults(
    dirName='/path/to/results',
    format='passes'
)
```

#### DOC Database Survey Export
```python
# Get survey info from user (via dialog or input)
survey_info = {
    'operator': 'Your Name',
    'easting': '1234567',
    'northing': '5678901',
    'recorder': 'SM4BAT',
    # ... other survey fields
}

bat_exporter.exportSurvey(
    dirName='/path/to/results',
    responses=survey_info,
    threshold1=0.85
)
```

## Example: Complete Workflow

### GUI Example

```python
from src.core.BatchProcessor import BatchProcessor, BatchProcessorCallbacks
from src.core.BatExporter import BatExporter
from src.core.BatDetector import BatDetector

# 1. Run batch processing
callbacks = MyGUICallbacks()
processor = BatchProcessor(
    configdir=configdir,
    directory=directory,
    recognisers=['NZ Bats'],
    callbacks=callbacks
)

result = processor.process_files()

if result == 0:
    # 2. Processing succeeded - now export
    bat_detector = BatDetector()
    bat_exporter = BatExporter(processor.config, bat_detector)
    
    # Export to XML
    bat_exporter.exportResults(directory, format='xml')
    
    # Optionally: export to DOC database
    survey_info = show_survey_dialog()  # Your UI dialog
    if survey_info:
        bat_exporter.exportSurvey(directory, survey_info)
```

### CLI Example

```python
from src.core.BatchProcessor import BatchProcessor
from src.core.BatExporter import BatExporter
from src.core.BatDetector import BatDetector

# 1. Run processing
processor = BatchProcessor(
    configdir=configdir,
    directory=directory,
    recognisers=['NZ Bats'],
    callbacks=cli_callbacks
)

result = processor.process_files()

if result == 0:
    # 2. Export results
    bat_detector = BatDetector()
    bat_exporter = BatExporter(processor.config, bat_detector)
    
    # Export formats as needed
    bat_exporter.exportResults(directory, format='xml')
    bat_exporter.exportResults(directory, format='csv')
```

## Benefits of This Approach

1. **Separation of Concerns**: BatchProcessor focuses on processing, BatExporter handles exporting
2. **Flexibility**: UI can choose when and how to export, separate from processing
3. **Consistency**: Matches bird workflow (birds don't have export in BatchProcessor either)
4. **Testability**: Each component can be tested independently
5. **Clearer Dependencies**: Export is an optional step after processing

## Migration Checklist

- [x] Remove export methods from BatchProcessor
- [x] Remove BatExporter import and field from BatchProcessor
- [x] Remove get_bat_survey_info callback
- [x] Update UI code to use BatExporter directly
- [ ] Test bat processing and export workflow in GUI
- [ ] Test bat processing and export workflow in CLI
- [ ] Update any documentation that references old export methods
