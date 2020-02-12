All notable changes to AviaNZ program will be documented in this file.

## [Unreleased]


## [2.1] - 2020-02-12

### Added

- Intermittent sampling option (under 'Action' menu)
- Recent files


## [2.0.2] - 2020-01-16

### Added

- Splitting the extra long segments made by 'Any sound' mode (Batch Processing)

### Changed

- 'Skip if certainty above' default to 90 for clarity (earlier 100; Review Batch Results)


## [2.0.1] - 2019-11-26

### Fixed .WAV


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

- Filter format and segment format to allow call types and certainty [@file_format_specification.md] (https://github.com/smarsland/AviaNZ/blob/master/Docs/file_format_specification.md)
- Windows setup as a wizard
- Mac setup automated with a wrapper


[unreleased]: https://github.com/smarsland/AviaNZ/compare/v2.0...HEAD
[2.1]: : https://github.com/smarsland/AviaNZ/compare/v2.0.2...v2.1
[2.0.2]: https://github.com/smarsland/AviaNZ/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/smarsland/AviaNZ/compare/v2.0...v2.0.1
[2.0]: https://github.com/smarsland/AviaNZ/compare/v1.5.1...v2.0