# AviaNZ Manual - Qt6 Refactoring Plan

**Goal**: Break the 6,124-line `AviaNZ_manual.py` into 3-4 manageable modules following Qt6 patterns.

## Core Principles
- **QMainWindow stays as UI controller** - Don't extract menu/frame/widget creation
- **Extract business logic only** - File I/O, audio processing, data operations  
- **Model-View separation** - Managers handle data, main window handles UI
- **Test after each extraction** - Ensure identical functionality

## Architecture Overview
```
AviaNZ_manual.py (3200 lines)     ← UI creation, events, signal coordination, mouse handling
├── audio_file_manager.py (800)   ← File I/O, loading, validation, navigation
├── segment_manager.py (700)      ← Segment data operations, overview management
├── audio_processor.py (500)      ← Signal processing, coordinate conversions
├── playback_manager.py (400)     ← Media playback control, audio output
├── config_manager.py (400)       ← Configuration handling, settings persistence
└── display_manager.py (500)      ← Graphics rendering, spectrogram display
```

## Communication Pattern
```python
# Business logic modules emit specific signals, main window handles UI updates
self.file_manager.file_loaded.connect(self.on_file_loaded)
self.file_manager.file_list_updated.connect(self.refresh_file_list_display)
self.segment_manager.segment_added.connect(self.draw_new_segment_graphics)
self.segment_manager.segment_selection_changed.connect(self.update_segment_controls)
self.segment_manager.segment_labels_updated.connect(self.refresh_segment_display)
self.audio_processor.processing_started.connect(self.update_status_message)
self.audio_processor.processing_completed.connect(self.refresh_displays)
self.audio_processor.coordinates_converted.connect(self.update_coordinate_displays)
self.playback_manager.playback_started.connect(self.update_play_button_state)
self.playback_manager.playback_position_changed.connect(self.move_playback_slider)
self.playback_manager.volume_changed.connect(self.update_volume_display)
self.config_manager.settings_changed.connect(self.apply_interface_changes)
self.display_manager.spectrogram_ready.connect(self.update_spectrogram_view)
```

## Implementation Order
1. ConfigManager (no dependencies)
2. AudioProcessor (signal processing, coordinate conversions)
3. PlaybackManager (media control, uses config manager)
4. DisplayManager (graphics rendering, uses config manager)
5. AudioFileManager (uses config manager, audio processor)
6. SegmentManager (uses file manager, display manager)
7. Main Window cleanup

## Rollback Safety
- Backup original file before starting
- Each step is independently testable
- Can revert individual modules if needed


# Refactoring Checklist

## □ **Phase 1: Extract ConfigManager**

### □ Setup
- [ ] `cp AviaNZ_manual.py AviaNZ_manual_backup.py`
- [ ] `touch config_manager.py`

### □ Create Module
- [ ] Copy interface template with signals (`settings_changed`, `config_loaded`, `config_saved`)
- [ ] Move these methods:
  - [ ] `changeSettings()` (~5600) → `show_settings_dialog()`
  - [ ] `changeParams()` (~5700) → `update_setting()` + emit signals
  - [ ] Config file loading/saving logic from `__init__()` and `closeFile()`
  - [ ] Parameter tree creation and management
  - [ ] Bird list file management

### □ Update Main Window
- [ ] Import ConfigManager
- [ ] Instantiate in `__init__()`: `self.config_manager = ConfigManager(self.configdir)`
- [ ] Connect signals: `self.config_manager.settings_changed.connect(self.apply_setting_change)`
- [ ] Update menu handlers to delegate: `self.config_manager.show_settings_dialog()`

### □ Test
- [ ] **Test configuration loading works**
- [ ] **Test settings dialog works**
- [ ] **Test parameter changes apply correctly**

## □ **Phase 2: Extract AudioProcessor**

### □ Setup
- [ ] `touch audio_processor.py`

### □ Create Module
- [ ] Copy interface template with signals (`processing_started`, `processing_completed`, `coordinates_converted`)
- [ ] Move these methods:
  - [ ] `convertAmpltoSpec()`, `convertSpectoAmpl()` (lines 1975-2005) → coordinate conversion methods
  - [ ] `convertYtoFreq()`, `convertFreqtoY()` → frequency conversion methods
  - [ ] `convertMillisecs()` → time conversion method
  - [ ] `denoise()` (line 4125) → `denoise_full_audio()` - remove progress dialogs
  - [ ] `denoiseSeg()` → `denoise_segment()`
  - [ ] `decomposeWP()` → `decompose_wavelet_packet()`
  - [ ] Signal processing utility functions

### □ Update Main Window
- [ ] Import AudioProcessor
- [ ] Instantiate: `self.audio_processor = AudioProcessor(self.config_manager)`
- [ ] Connect signals: 
  - [ ] `self.audio_processor.processing_started.connect(self.update_status_message)`
  - [ ] `self.audio_processor.processing_completed.connect(self.refresh_displays)`
- [ ] Update handlers to delegate coordinate conversions and signal processing

### □ Test
- [ ] **Test coordinate conversions work identically**
- [ ] **Test audio processing works**
- [ ] **Test denoising works**

## □ **Phase 3: Extract PlaybackManager**

### □ Create Module
- [ ] `touch playback_manager.py`
- [ ] Copy interface template with signals (`playback_started`, `playback_stopped`, `playback_position_changed`, `volume_changed`)
- [ ] Move these methods:
  - [ ] `playVisible()` (line 5308) → `start_playback_visible()` - remove UI updates
  - [ ] `playSelectedSegment()` → `start_playback_segment()`
  - [ ] `playBandLimitedSegment()` → `start_playback_band_limited()`
  - [ ] `pausePlayback()` → `pause_playback()`
  - [ ] `stopPlayback()` → `stop_playback()`
  - [ ] `setSpeed()` (line 2193) → `set_playback_speed()`
  - [ ] `movePlaySlider()` → `update_playback_position()`
  - [ ] `volSliderMoved()` → `set_volume()`
  - [ ] `barMoved()` → `seek_to_position()`

### □ Update Main Window
- [ ] Import PlaybackManager
- [ ] Instantiate: `self.playback_manager = PlaybackManager(self.config_manager)`
- [ ] Set media object: `self.playback_manager.set_media_object(self.media_obj)`
- [ ] Connect signals: 
  - [ ] `self.playback_manager.playback_started.connect(self.update_play_button_state)`
  - [ ] `self.playback_manager.playback_position_changed.connect(self.move_playback_slider)`
  - [ ] `self.playback_manager.volume_changed.connect(self.update_volume_display)`
- [ ] Update handlers to delegate: `self.playback_manager.start_playback_visible(start_ms, end_ms)`

### □ Test
- [ ] **Test playback controls work identically**
- [ ] **Test volume controls work**
- [ ] **Test playback position tracking works**

## □ **Phase 4: Extract DisplayManager**

### □ Create Module
- [ ] `touch display_manager.py`
- [ ] Copy interface template with signals (`spectrogram_ready`, `overview_updated`, `graphics_refreshed`)
- [ ] Move these methods:
  - [ ] `drawfigMain()` (line ~2100) → `render_main_displays()` - graphics rendering only
  - [ ] `drawOverview()` (line ~2020) → `render_overview()`
  - [ ] `setfigs()` (line ~2090) → `update_figure_data()`
  - [ ] `setSpectrogram()` → `apply_spectrogram_settings()`
  - [ ] `setColourMap()`, `setColourLevels()` → color/contrast management
  - [ ] `drawGuidelines()` → `render_frequency_guides()`
  - [ ] `drawProtocolMarks()` → `render_protocol_marks()`

### □ Update Main Window
- [ ] Import DisplayManager
- [ ] Instantiate: `self.display_manager = DisplayManager(self.config_manager)`
- [ ] Connect signals: `self.display_manager.spectrogram_ready.connect(self.on_spectrogram_updated)`
- [ ] Update rendering calls to delegate to display manager

### □ Test
- [ ] **Test spectrogram rendering works**
- [ ] **Test overview display works**
- [ ] **Test color/contrast controls work**

## □ **Phase 6: Extract SegmentManager**

### □ Create Module
- [ ] `touch segment_manager.py`
- [ ] Copy interface template with signals (`segment_added`, `segment_deleted`, `segment_updated`, `segment_selection_changed`, `segment_labels_updated`)
- [ ] Move these methods:
  - [ ] `addSegment()` (line 2677) → `create_segment()` - data operations only
  - [ ] `deleteSegment()` (~5900) → `delete_segment()`
  - [ ] `selectSegment()` (line 2834) → `select_segment()` - selection logic only
  - [ ] `deselectSegment()` → `deselect_segment()`
  - [ ] `confirmSegment()` → `confirm_segment_labels()`
  - [ ] `addRegularSegments()` → `add_regular_segments()`
  - [ ] `refreshOverviewWith()` (line 2627) → `update_overview_display()`
  - [ ] `updateRegion_spec()`, `updateRegion_ampl()` → `update_segment_bounds()` - data only
  - [ ] `refreshSegmentControls()` → `update_segment_ui_state()`

### □ Update Main Window  
- [ ] Import SegmentManager
- [ ] Instantiate: `self.segment_manager = SegmentManager(self.config_manager, self.file_manager, self.display_manager)`
- [ ] Connect signals: 
  - [ ] `self.segment_manager.segment_added.connect(self.draw_new_segment_graphics)`
  - [ ] `self.segment_manager.segment_selection_changed.connect(self.update_segment_controls)`
  - [ ] `self.segment_manager.segment_labels_updated.connect(self.refresh_segment_ui)`
- [ ] Update mouse handlers to delegate: `self.segment_manager.create_segment(start, end, low_freq, high_freq)`
- [ ] Keep graphics creation in UI event handlers

### □ Test
- [ ] **Test segment creation works identically**
- [ ] **Test segment deletion works**
- [ ] **Test segment selection works**
- [ ] **Test segment editing works**

## □ **Phase 7: Main Window Cleanup**

### □ Core Responsibilities (What Remains)
- [ ] **UI Layout and Widget Creation** - Qt window setup, menus, toolbars, splitters
- [ ] **Mouse and Keyboard Event Handling** - coordinate to manager mapping only
- [ ] **Signal Coordination** - connecting manager signals to UI updates  
- [ ] **Graphics Rendering Coordination** - pyqtgraph drawing calls

### □ Remove Extracted Code
- [ ] Delete original methods that were moved to managers
- [ ] Keep any compatibility stubs needed for external calls
- [ ] Update imports to reflect new module structure

### □ Architecture Validation
- [ ] **Verify Manager Dependencies:**
  - [ ] ConfigManager (independent)
  - [ ] AudioProcessor (needs ConfigManager)
  - [ ] PlaybackManager (needs ConfigManager, AudioProcessor)
  - [ ] DisplayManager (needs ConfigManager, AudioProcessor) 
  - [ ] AudioFileManager (needs ConfigManager, AudioProcessor)
  - [ ] SegmentManager (needs ConfigManager, AudioFileManager, DisplayManager)

### □ Final Testing
- [ ] **Complete feature test:** Load file → View spectrogram → Create segments → Play audio → Save
- [ ] **Test manager isolation:** Modify one manager without affecting others
- [ ] **Test Qt integration:** UI responsiveness maintained
- [ ] **Memory and performance:** No significant degradation

### □ Documentation
- [ ] Add docstrings to new modules
- [ ] Update main window comments
- [ ] Document signal connections

## □ **Cleanup**
- [ ] Remove backup file if everything works
- [ ] Commit changes to git
- [ ] Update requirements.txt if needed
