All notable changes to AviaNZ program will be documented in this file.

## [Unreleased]

### Fixed

- Empty results in Recogniser testing
- Cluster button scale
- Better feedback in Batch Processing and Review

### Added

- CNN training to improve precision
- Call type review in any sound (all species) mode
- Call type annotation/correction in manual processing mode
- Bach mode progress dialog
- Time axis in single species review
- Recording information in the manual processing mode

- Bat mode: loading, annotating and batch-processing DoC format bitmaps, with file-level annotations
- NZ bat list and filter to identify them (click detector and CNN)

### Changed

- Removed user input for number of thresholds in wavelet filter training
- Filter requirements are more flexible now, to incorporate bat settings

## [2.2] - 2020-04-28

### Fixed

- Recogniser testing summary
- Storing zoom levels between All Species review runs
- Better zoom and scaling in review dialogs
- Faster and safer Excel export
- Splitter outputs duration of existing .data files properly
- Dialogs adapted to low screen resolutions
- Various minor bug fixes

### Added

- CNN layer after wavelet filter in the processing pipeline to improve the precision
- Certainty improved after CNN post-proc
- Trained CNN models for NI brown kiwi, ready to use
- Separate excel export in Batch Review
- Intermittent sampling
- Single Species review now has a frequency axis
- All Species review now has a "Question" button to mark segments that need editing
- File list in Manual mode now has colored icons to identify what level of annotations is present in each file

### Changed

- Human review made faster by loading the segments rather than reading whole file
- Single Species review now displays every piece of long segments individually, rather than just showing the center
- Lots of visual updates

## [2.1] - 2020-02-12

### Added

- Intermittent sampling option (under 'Action' menu)
- Recent files
- Split the extra long segments made by 'Any sound' mode (Batch Processing)

### Changed

- 'Skip if certainty above' default to 90 for clarity (earlier 100; Review Batch Results)

### Fixed

 - .WAV case-insensitive

## [2.0] - 2019-11-22

### Added

- Call types (segments and wavelet filters)
- Clustering (wavelet filter training)
- Detailed segment information (Manual Processing)
- Option to on/off wind detection (Batch Processing)
- Dialog for filter (recogniser) management, export and import
- Slow and fast playback button (Manual Processing)
- Paging to deal with memory for large files (Batch Processing)
- Option to run several filters in a single pass, saving time
- Separate wizards for filter training and testing
- "Toggle all" option (single species review)
- WAV+DATA splitter

### Changed

- Filter format and segment format to allow call types and certainty (file_format_specification.md)
- Windows setup as a wizard
- Mac setup automated with a wrapper

## [1.5.1] - 2019-08-13

### Added

- Spectrogram zoom buttons for All Species review

### Fixed

- Various bug fixes after the workshop

## [1.5] - 2019-08-05

### Added

- Option to include wind masking (Batch mode)
- Impulse masking on file read for auto detection

### Changed

- Faster filter training

### Fixed

- Minor bugs

## [1.4] - 2019-06-10

### Added

- Online version of Cheat Sheet
- Users can now train their own wavelet filters
- New feature - spectral derivatives
- "Quick Denoise" button

### Changed

- Wavelet packet decomposition and reconstruction rewritten entirely, allows partial (faster) and full (slower) anti-aliasing
- Various small improvements to bird list selection and custom species input

### Fixed

- Single Species review
- Lots of other minor bugs

## [1.3] - 2018-10-24

### Added

-

### Changed

- AviaNZ config, bird lists, and filter files to the home directory; in C: /Users/ username/ AppData/Roaming/ AviaNZ for Windows and ~/ .avianz for Mac and Linux

### Fixed

- Various bugs

## [1.2] - 2018-10-24

### Fixed

- Various bugs

## [1.1] - 2018-09-16   **

### Added

- Paging in the main interface
- Volume controller
- Option to playback only the target frequency band of a segment
- Operator and Reviewer
- Status bar (status updates and operator/reviewer)
- 'Quit' to the menu
- 'Make read only' option to avoid segments by mistake
- Interface Settings (under the Appearance Menu) to customise the annotation colour scheme, length of the annotation overview sections, auto save frequency, page size etc.
- 'Delete all segments'
- Export segments to excel (Manual Segmentation)
- Save spectrogram as image
- Save selected sound
- Help menu pointing to the manual
- Batch Processing (all species/target species)
- Review Batch Results
- Excel summary for the batch mode

### Changed

- Start interface now has three options: Manual Segmentation, Batch Processing, and Review Batch Results
- Assigned left/right mouse buttons for performing segmentation (one button to select and edit a segment, and the other to create a segment; user can choose which performs which action)
- Two options to Human Review (All segments/Choose species)

### Fixed

- Various bugs

## [1.0] - 2018-03-29   **

### Fixed

- Various bugs

## [0.1] - 2017-06-08

Initial Release (for the Acoustic Workshop at Massey University 2017-06-09)

### Added

- Manual annotation of field recordings (with different certainty - color code)
- Option to add segments as boxes and full-height start-end marks
- Auto saving of the segments
- Zooming and scrolling
- Sound playback options
- Overview spectrogram image
- Fundamental frequency
- Option to change spectrogram parameters
- Denoise and band-pass filtering options
- Automated segmentation (Wavelets, median clipping, fundamental frequency, etc.)
- Find Matches (select a segment and then the program looks for others that are similar. It doesn't currently deal well with nosie)
- Check segments (human review options)
- Spectrogram CheatSheet (pdf format)


[unreleased]: https://github.com/smarsland/AviaNZ/compare/v2.2...HEAD
[2.2]: https://github.com/smarsland/AviaNZ/compare/v2.1...v2.2
[2.1]: https://github.com/smarsland/AviaNZ/compare/v2.0.2...v2.1
[2.0]: https://github.com/smarsland/AviaNZ/compare/v1.5.1...v2.0
[1.5.1]: https://github.com/smarsland/AviaNZ/compare/v1.5...v1.5.1
[1.5]: https://github.com/smarsland/AviaNZ/compare/v1.4...v1.5
