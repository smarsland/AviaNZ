All notable changes to AviaNZ program will be documented in this file.

## Unreleased

### Added
- New changepoint detector replaces wavelet filter as the default non-NN recognition method (wavelet filters still work but considered deprecated)
- Syllable-level detection: trainable filters via the new changepoint pipeline, one example filter for LSK syllables included, and parameter ranges for non-specific detectors adapted
- New method for wind noise removal by polynomial fitting (OLS or robust options available)
- Pre-built GPU support for NN in compiled versions
- Option to loop playback in both review types
- Option to autoplay in one-by-one review
- Non-linear frequency scale spectrograms (Mel, Bark)
- Additional spectrogram normalization options, including PCEN

### Changed
- PyQt6 instead of PyQt5
- Training will now include subdirectories when searching for data
- Node selection in training faster, more stable, and produces ROCs more closely consistent with testing/processing
- Wind noise removal only available for changepoint detectors now
- Denoising dialog will now use time-adaptive noise estimation by default
- Filter format extended to allow setting segmenter
- Better UI for adding species in review, search function
- Shorter pages (5 mins) for low sampling rate files in batch mode
- NNs no longer redefine segment boundaries, only accept/reject
- reduced extension length when applying NNs to short segments
- various changes to NN training
- Tab shortcut for species/calltype switch
- different click detection process when training bat detectors
- ground truth files no longer store resolution in the name
- batch processing will not allow upsampling if using wind filter (only 2x or 4x "fake upsampling" which is implemented by node remapping)
- batch review now has better feedback on input errors

### Fixed
- Faster NN classification
- Post-processing harmonised between batch mode, testing, testing with NN
- Segmenting in manual mode now only overwrites the current page segments
- Problems when undoing segmentation in multi-page files
- Bat BMP reading sometimes wasn't setting a parameter
- Faster spectrogram updating in manual mode
- Minor UI bugs in recogniser training wizard
- Long species list now triggers button reordering in review
- better edge case handling when adding species in review
- ancient bugs in best basis selection for wavelet denoising
- WAV Splitter produced bad timestamps on files starting within an hour before a DST change
- data padding for wavelet decomposition/reconstruction was wrong length and reversed
- Wavelet energy computation made safer and less edge-influenced
- missing post-processing settings in batch GUI

### Hidden changes (developer-mode):
- Ridge/instantaneous freq. detection
- Shape analysis dialog
- Button for exporting segment sound at different speed
- Denoising button now connected to the new denoising
- Saving spectrogram images without axes
- Call comparator as a separate script, clock-adjustment only


## [3.2] - 2021-02-16

### Added
- Ability to undo previously deleted segments in review
- "One-by-one" review plot size expands to dialog size
- BatSearch-compatible output for batmode
- National Bat Database format output for batmode
- Segment saving feedback now shown in status bar
- Mouse cursor indicates current mode
- "Jump to next annotation" buttons
- Optional frequency masking in NN training
- Ability to customise existing recognisers
- Formant marking in spectrogram
- Dialog for adjusting thresholds of stored filters
- Ability to adjust other filter parameters for newly-trained detectors

### Changed
- Improved processing pipeline for bats
- Improved NN recogniser for bats
- Extended morepork recogniser with NN
- "One-by-one" review (previously All Species) can now be run on single species
- Batch Review settings separated into Advanced and Simple
- Simpler dropdown to set review certainty bounds
- Better spectrogram type selection UI
- Separate UI thread to keep responsiveness when batch processing

### Fixed
- Greatly reduced CPU load for mouseover detection in spectrogram items
- Batch settings are appropriately greyed out and provide tooltips
- 8-bit WAV playback was not working
- Bugs in non-specific batch processing
- Cleaned up gap-merging algorithms, could have caused bugs in edge cases

## [3.1] - 2020-10-09

### Added

- Visible frequency range in batmode defaults to full
- Manage Recognisers deals with NN files as well
- ROCs during training show more information, wavelet freq. bands

### Changed

- Excel output shows absolute times of detected segments, when files have timestamps
- Tensorflow version bumped up

### Fixed

- Graphical problems on high resolution displays
- Safer restarting

## [3.0] - 2020-09-17

### Added

- *NN training* to improve precision
- Call type annotation/correction in manual processing mode
- Call type review in all species mode

- *Bat mode:* loading, annotating and batch-processing DoC format bitmaps, with file-level annotations
- NZ bat list and a filter to identify them (click detector and NN)
- Frequency guides for fast bat call annotation

- *Utilities:* importing Freebird or Excel format annotations, backing up data files

- Command line options for batch processing and testing

- New filters for kiwi species
- New sound and spectrogram samples

- Recording information in the manual processing mode
- Time axis in single species review

### Changed

- Removed user input for number of thresholds in wavelet filter training
- Filter requirements more flexible now, to incorporate bat settings
- Improved overview window UI
- Better feedback, progress bars in Batch Processing and Review
- Customizable tile size in Single species review
- Easier access to WAV/DATA splitter
- Better presentation of recogniser testing results
- Wavelet training now reads clusters from calltype annotations if provided
- Removed fund. freq. option from wavelet filter training

### Fixed

- Avoiding div by 0 in edge cases of spectrogram normalization
- No more CTD on basic import errors in windows
- Some minor bugfixes and safety checks

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

- NN layer after wavelet filter in the processing pipeline to improve the precision
- Certainty improved after NN post-proc
- Trained NN models for NI brown kiwi, ready to use
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
