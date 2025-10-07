# Deprecated Bat Export Methods

## Overview
This file contains deprecated bat export methods that were removed from `AviaNZ_batch.py` as part of the refactoring effort to consolidate bat processing into a single unified interface.

## Date Moved
October 5, 2025

## Reason for Deprecation
These methods were replaced by the unified `exportBatResults()` method which supports multiple export formats through a single interface:
- `exportBatResults(dirName, format='xml')` - replaces all 3 versions of `exportToBatSearch`
- `exportBatResults(dirName, format='csv')` - replaces `exportToBatSearchCSV`
- `exportBatResults(dirName, format='passes')` - replaces `outputBatPasses`

## Deprecated Methods
1. **outputBatPasses()** - Export bat passes to CSV
2. **exportToBatSearch()** - Export to BatSearch XML format (using lxml)
3. **exportToBatSearch_1()** - Earlier version of BatSearch XML export
4. **exportToBatSearch_2()** - Another version of BatSearch XML export (string-based XML)
5. **exportToBatSearchCSV()** - Export to BatSearch-compatible CSV

## Impact
- The only active usage was in `/archive/run.py`, which is already archived
- All active code in the main application uses the new `exportBatResults()` method
- Comments in `batch_interface.py` referenced these methods but they were already commented out

## If You Need These Methods
If you have old scripts that use these methods:
1. Update your code to use `exportBatResults()` with the appropriate format parameter
2. Or copy the methods from `deprecated_bat_exports.py` into your script (not recommended)

## Active Bat Methods Retained
- `exportBatResults()` - Unified export method (active)
- `exportBatSurvey()` - DOC database export (active, used by `exportToDOCDB()`)
- `processBatFile()` - Main bat processing method (active)
- `ClickSearch()` - Click detection (active)
- `labelBatFile()` - NN classification (active)
- Helper methods: `_exportBatXML()`, `_exportBatCSV()`, `_exportBatPasses()`, `_getBatLabel()`, `_parseTimeDate()`
