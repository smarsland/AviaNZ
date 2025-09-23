# Config Manager Module
# Extracted from AviaNZ_manual.py as part of Qt6 refactoring

import os
import json
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtGui import QColor
from pyqtgraph.parametertree import Parameter, ParameterTree
from ..core import SupportClasses
import re


class ConfigManager(QObject):
    """Handles configuration loading, saving, and settings dialog management."""
    
    # Signals emitted when configuration changes
    settings_changed = pyqtSignal(str, object)  # (setting_name, new_value)
    config_loaded = pyqtSignal()
    config_saved = pyqtSignal()
    
    def __init__(self, configdir, parent=None):
        super().__init__(parent)
        self.configdir = configdir
        self.configfile = os.path.join(configdir, "AviaNZconfig.txt")
        self.parent_window = parent
        
        # Initialize config loader
        self.ConfigLoader = SupportClasses.ConfigLoader()
        self.config = None
        
        # Parameter tree for settings dialog
        self.parameter_tree = None
        self.parameter_widget = None
        
        # Load configuration
        self.load_config()
        print("Config loaded")
        
    def load_config(self):
        """Load configuration from file."""
        self.config = self.ConfigLoader.config(self.configfile)
        
        # Ensure critical config values have correct types and defaults
        if 'specMouseAction' in self.config:
            try:
                # Handle empty string or invalid values
                if self.config['specMouseAction'] == '' or self.config['specMouseAction'] is None:
                    self.config['specMouseAction'] = 3  # Default to "Mark boxes by dragging"
                else:
                    self.config['specMouseAction'] = int(self.config['specMouseAction'])
            except (ValueError, TypeError):
                # If conversion fails, use default
                self.config['specMouseAction'] = 3
        else:
            # If key doesn't exist, add default
            self.config['specMouseAction'] = 3
        
        # Load filters
        self.filtersDir = os.path.join(self.configdir, self.config['FiltersDir'])
        self.FilterDicts = self.ConfigLoader.filters(self.filtersDir)
        
        # Load bird lists
        self.shortBirdList = self.ConfigLoader.shortbl(self.config['BirdListShort'], self.configdir)
        self.longBirdList = self.ConfigLoader.longbl(self.config['BirdListLong'], self.configdir)
        self.knownCalls = self.ConfigLoader.knownCalls(self.config['KnownCallsList'], self.configdir)
        self.batList = self.ConfigLoader.batl(self.config['BatList'], self.configdir)
        
        # Clean encoding issues in loaded lists
        self.shortBirdList = self._clean_species_list(self.shortBirdList)
        self.longBirdList = self._clean_species_list(self.longBirdList)
        self.batList = self._clean_species_list(self.batList)
        
        # Search filters for additional known calls
        if self.knownCalls is not None:
            for filt in self.FilterDicts.values():
                if not filt["species"] in self.knownCalls:
                    self.knownCalls[filt["species"]] = []
                for subf in filt["Filters"]:
                    if (not subf["calltype"] in self.knownCalls[filt["species"]] and 
                        not subf["calltype"] == "Not Specified" and 
                        not subf["calltype"] == "Add"):
                        self.knownCalls[filt["species"]].append(subf["calltype"])
        
        self.config_loaded.emit()
        
    def _clean_species_list(self, species_list):
        """Clean a species list to fix encoding issues."""
        if not species_list:
            return species_list
            
        cleaned_list = []
        for species in species_list:
            if isinstance(species, str):
                # Fix common encoding issues with apostrophes
                cleaned_species = species.replace('â', "'")  # Fix corrupted apostrophe
                cleaned_species = cleaned_species.replace(''', "'")  # Fix smart quote
                cleaned_species = cleaned_species.replace(''', "'")  # Fix another smart quote variant
                cleaned_list.append(cleaned_species)
            else:
                cleaned_list.append(species)
                
        return cleaned_list
        
    def show_settings_dialog(self):
        """Create and show the parameter tree settings dialog."""
        if self.parent_window:
            self.parent_window.saveSegments()
            
        # Get basenames for display
        fn1 = self.config['BirdListShort']
        if '/' in fn1:
            fn1 = os.path.basename(fn1)
        fn2 = self.config['BirdListLong']
        if fn2 is not None and '/' in fn2:
            fn2 = os.path.basename(fn2)
        fn3 = self.config['KnownCallsList']
        if fn3 is not None and '/' in fn3:
            fn3 = os.path.basename(fn3)
        fn4 = self.config['BatList']
        if fn4 is not None and '/' in fn4:
            fn4 = os.path.basename(fn4)
        fn5 = self.config['FreebirdList']
        if fn5 is not None and '/' in fn5:
            fn5 = os.path.basename(fn5)
            
        # Check for multiple segments (needed for readonly logic)
        hasMultipleSegments = False
        if hasattr(self.parent_window, 'segments'):
            for s in self.parent_window.segments:
                if len(s[4]) > 1:
                    hasMultipleSegments = True
                    break

        params = [
            {'name': 'Mouse settings', 'type': 'group', 'children': [
                {'name': 'Use right button to make segments', 'type': 'bool', 
                 'tip': 'If true, segments are drawn with right clicking.',
                 'value': self.config['drawingRightBtn']},
                {'name': 'Spectrogram mouse action', 'type': 'list', 
                 'values': ['Mark segments by clicking', 'Mark boxes by clicking', 'Mark boxes by dragging'],
                 'value': {1: 'Mark segments by clicking', 2: 'Mark boxes by clicking', 3: 'Mark boxes by dragging'}.get(self.config.get('specMouseAction', 3), 'Mark boxes by dragging')}
            ]},

            {'name': 'Paging', 'type': 'group', 'children': [
                {'name': 'Page size', 'type': 'float', 'value': self.config['maxFileShow'], 'limits': (5, 3600),
                 'step': 5, 'suffix': ' sec'},
                {'name': 'Page overlap', 'type': 'float', 'value': self.config['fileOverlap'], 'limits': (0, 20),
                 'step': 2, 'suffix': ' sec'},
            ]},

            {'name': 'Annotation', 'type': 'group', 'children': [
                {'name': 'Annotation overview cell length', 'type': 'float',
                 'value': self.config['widthOverviewSegment'],
                 'limits': (5, 300), 'step': 5, 'suffix': ' sec'},
                {'name': 'Make boxes transparent', 'type': 'bool',
                 'value': self.config['transparentBoxes']},
                {'name': 'Auto save segments every', 'type': 'float', 'value': self.config['secsSave'],
                 'step': 5, 'limits': (5, 900), 'suffix': ' sec'},
                {'name': 'Segment colours', 'type': 'group', 'children': [
                    {'name': 'Confirmed segments', 'type': 'color', 'value': self.config['ColourNamed'],
                     'tip': "Correctly labeled segments"},
                    {'name': 'Possible', 'type': 'color', 'value': self.config['ColourPossible'],
                     'tip': "Segments that need further approval"},
                    {'name': "Don't Know", 'type': 'color', 'value': self.config['ColourNone'],
                     'tip': "Segments that are not labelled"},
                    {'name': 'Currently selected', 'type': 'color', 'value': self.config['ColourSelected'],
                     'tip': "Currently selected segment"},
                ]},
                {'name': 'Guidelines', 'type': 'group', 'children': [
                    {'name': 'Show frequency guides', 'type': 'list', 'values':
                        {'Always': 'always', 'For bats only': 'bat', 'Never': 'never'},
                        'value': self.config['guidelinesOn']},
                    {'name': 'Guideline 1 frequency', 'type': 'float', 'value': self.config['guidepos'][0]/1000, 'limits': (0, 1000), 'suffix': ' kHz'},
                    {'name': 'Guideline 1 colour', 'type': 'color', 'value': self.config['guidecol'][0]},
                    {'name': 'Guideline 2 frequency', 'type': 'float', 'value': self.config['guidepos'][1]/1000, 'limits': (0, 1000), 'suffix': ' kHz'},
                    {'name': 'Guideline 2 colour', 'type': 'color', 'value': self.config['guidecol'][1]},
                    {'name': 'Guideline 3 frequency', 'type': 'float', 'value': self.config['guidepos'][2]/1000, 'limits': (0, 1000), 'suffix': ' kHz'},
                    {'name': 'Guideline 3 colour', 'type': 'color', 'value': self.config['guidecol'][2]},
                    {'name': 'Guideline 4 frequency', 'type': 'float', 'value': self.config['guidepos'][3]/1000, 'limits': (0, 1000), 'suffix': ' kHz'},
                    {'name': 'Guideline 4 colour', 'type': 'color', 'value': self.config['guidecol'][3]},
                ]},
                {'name': 'Check-ignore protocol', 'type': 'group', 'children': [
                    {'name': 'Show check-ignore marks', 'type': 'bool', 'value': self.config['protocolOn']},
                    {'name': 'Length of checking zone', 'type': 'float', 'value': self.config['protocolSize'],
                     'limits': (1, 300), 'step': 1, 'suffix': ' sec'},
                    {'name': 'Repeat zones every', 'type': 'float', 'value': self.config['protocolInterval'],
                     'limits': (1, 300), 'step': 1, 'suffix': ' sec'},
                    {'name': 'Line colour', 'type': 'color', 'value': self.config['protocolLineCol']},
                    {'name': 'Line width', 'type': 'int', 'value': self.config['protocolLineWidth'],
                     'limits': (1, 10), 'step': 1},
                ]}
            ]},

            {'name': 'Bird List', 'type': 'group', 'children': [
                {'name': 'Common Bird List', 'type': 'group', 'children': [
                    {'name': 'Filename', 'type': 'str', 'value': fn1, 'readonly': True},
                    {'name': 'Choose File', 'type': 'action'},
                ]},
                {'name': 'Full Bird List', 'type': 'group', 'children': [
                    {'name': 'Filename', 'type': 'str', 'value': fn2, 'readonly': True},
                    {'name': 'Choose File', 'type': 'action'}
                ]},
                {'name': 'Known Calls List', 'type': 'group', 'children': [
                    {'name': 'Filename', 'type': 'str', 'value': fn3, 'readonly': True},
                    {'name': 'Choose File', 'type': 'action'},
                ]},
                {'name': 'Bat List', 'type': 'group', 'children': [
                    {'name': 'Filename', 'type': 'str', 'value': fn4, 'readonly': True},
                    {'name': 'Choose File', 'type': 'action'}
                ]},
                {'name': 'Freebird List', 'type': 'group', 'children': [
                    {'name': 'Filename', 'type': 'str', 'value': fn5, 'readonly': True},
                    {'name': 'Choose File', 'type': 'action'}
                ]},
                {'name': 'Dynamically reorder bird list', 'type': 'bool', 'value': self.config['ReorderList']},
                {'name': 'Default to multiple species', 'type': 'bool', 'value': self.config['MultipleSpecies'],
                 'readonly': hasMultipleSegments},
                {'name': 'Include calltype', 'type': 'bool', 'value': self.config['IncludeCalltype']},
            ]},
            {'name': 'User', 'type': 'group', 'children': [
                {'name': 'Operator', 'type': 'str', 'value': self.config['operator'],
                 'tip': "Person name"},
                {'name': 'Reviewer', 'type': 'str', 'value': self.config['reviewer'],
                 'tip': "Person name"},
            ]},
            {'name': 'Maximise window on startup', 'type': 'bool', 'value': self.config['StartMaximized']},
            {'name': 'Require noise data', 'type': 'bool', 'value': self.config['RequireNoiseData']},
        ]

        # Create tree of Parameter objects
        self.parameter_tree = Parameter.create(name='params', type='group', children=params)
        self.parameter_tree.sigTreeStateChanged.connect(self.update_setting)
        
        # Create ParameterTree widget
        self.parameter_widget = ParameterTree()
        self.parameter_widget.setParameters(self.parameter_tree, showTop=False)
        self.parameter_widget.show()
        self.parameter_widget.setWindowTitle('AviaNZ - Interface Settings')
        self.parameter_widget.setWindowIcon(self.parent_window.windowIcon() if self.parent_window else None)
        self.parameter_widget.setFixedHeight(900)
        self.parameter_widget.setMinimumWidth(520)

    def update_setting(self, param, changes):
        """Update configuration when parameter tree changes."""
        # Save segments first if parent window exists
        if self.parent_window and hasattr(self.parent_window, 'saveSegments'):
            self.parent_window.saveSegments()

        # Regexes to parse guideline settings
        rgx_guide_pos = re.compile(r"Annotation.Guidelines.Guideline ([0-9]) frequency")
        rgx_guide_col = re.compile(r"Annotation.Guidelines.Guideline ([0-9]) colour")

        for param, change, data in changes:
            path = self.parameter_tree.childPath(param)
            if path is not None:
                childName = '.'.join(path)
            else:
                childName = param.name()

            # Handle different setting changes
            if childName == 'Output parameters.Auto save segments every':
                self.config['secsSave'] = data
                self.settings_changed.emit('secsSave', data)
            elif childName == 'Annotation.Annotation overview cell length':
                self.config['widthOverviewSegment'] = data
                self.settings_changed.emit('widthOverviewSegment', data)
            elif childName == 'Annotation.Make boxes transparent':
                self.config['transparentBoxes'] = data
                self.settings_changed.emit('transparentBoxes', data)
            elif childName == 'Mouse settings.Use right button to make segments':
                self.config['drawingRightBtn'] = data
                self.settings_changed.emit('drawingRightBtn', data)
            elif childName == 'Mouse settings.Spectrogram mouse action':
                # Map the string selection back to integer
                action_map = {
                    'Mark segments by clicking': 1, 
                    'Mark boxes by clicking': 2, 
                    'Mark boxes by dragging': 3
                }
                
                if data in action_map:
                    data = action_map[data]
                elif isinstance(data, str) and data.strip() == '':
                    # Handle empty string case
                    data = 3  # Default value
                else:
                    # Try to convert to int, or use default
                    try:
                        data = int(data)
                        if data not in [1, 2, 3]:
                            data = 3
                    except (ValueError, TypeError):
                        data = 3
                    
                self.config['specMouseAction'] = data
                self.settings_changed.emit('specMouseAction', data)
            elif childName == 'Paging.Page size':
                self.config['maxFileShow'] = data
                self.settings_changed.emit('maxFileShow', data)
            elif childName == 'Paging.Page overlap':
                self.config['fileOverlap'] = data
                self.settings_changed.emit('fileOverlap', data)
            elif childName == 'Maximise window on startup':
                self.config['StartMaximized'] = data
                self.settings_changed.emit('StartMaximized', data)
            elif childName == 'Bird List.Dynamically reorder bird list':
                self.config['ReorderList'] = data
                self.settings_changed.emit('ReorderList', data)
            elif childName == 'Bird List.Default to multiple species':
                self.config['MultipleSpecies'] = data
                self.settings_changed.emit('MultipleSpecies', data)
            elif childName == 'Bird List.Include calltype':
                self.config['IncludeCalltype'] = data
                self.settings_changed.emit('IncludeCalltype', data)
            elif childName == 'Require noise data':
                self.config['RequireNoiseData'] = data
                self.settings_changed.emit('RequireNoiseData', data)
            elif childName == 'User.Operator':
                self.config['operator'] = data
                self.settings_changed.emit('operator', data)
            elif childName == 'User.Reviewer':
                self.config['reviewer'] = data
                self.settings_changed.emit('reviewer', data)
            elif childName == 'Annotation.Segment colours.Confirmed segments':
                rgbaNamed = list(data.getRgb())
                if rgbaNamed[3] > 100:
                    rgbaNamed[3] = 100
                self.config['ColourNamed'] = rgbaNamed
                self.settings_changed.emit('ColourNamed', rgbaNamed)
            elif childName == 'Annotation.Segment colours.Possible':
                rgbaVal = list(data.getRgb())
                if rgbaVal[3] > 100:
                    rgbaVal[3] = 100
                self.config['ColourPossible'] = rgbaVal
                self.settings_changed.emit('ColourPossible', rgbaVal)
            elif childName == "Annotation.Segment colours.Don't Know":
                rgbaVal = list(data.getRgb())
                if rgbaVal[3] > 100:
                    rgbaVal[3] = 100
                self.config['ColourNone'] = rgbaVal
                self.settings_changed.emit('ColourNone', rgbaVal)
            elif childName == 'Annotation.Segment colours.Currently selected':
                rgbaVal = list(data.getRgb())
                if rgbaVal[3] > 100:
                    rgbaVal[3] = 100
                self.config['ColourSelected'] = rgbaVal
                self.settings_changed.emit('ColourSelected', rgbaVal)
            elif childName == 'Annotation.Guidelines.Show frequency guides':
                self.config['guidelinesOn'] = data
                self.settings_changed.emit('guidelinesOn', data)
            elif rgx_guide_pos.match(childName):
                guideid = int(rgx_guide_pos.search(childName).group(1)) - 1
                self.config['guidepos'][guideid] = float(data) * 1000
                self.settings_changed.emit('guidepos', self.config['guidepos'])
            elif rgx_guide_col.match(childName):
                guideid = int(rgx_guide_col.search(childName).group(1)) - 1
                self.config['guidecol'][guideid] = data
                self.settings_changed.emit('guidecol', self.config['guidecol'])
            elif childName == 'Annotation.Check-ignore protocol.Show check-ignore marks':
                self.config['protocolOn'] = data
                self.settings_changed.emit('protocolOn', data)
            elif childName == 'Annotation.Check-ignore protocol.Length of checking zone':
                self.config['protocolSize'] = data
                self.settings_changed.emit('protocolSize', data)
            elif childName == 'Annotation.Check-ignore protocol.Repeat zones every':
                self.config['protocolInterval'] = data
                self.settings_changed.emit('protocolInterval', data)
            elif childName == 'Annotation.Check-ignore protocol.Line colour':
                rgbaVal = list(data.getRgb())
                self.config['protocolLineCol'] = rgbaVal
                self.settings_changed.emit('protocolLineCol', rgbaVal)
            elif childName == 'Annotation.Check-ignore protocol.Line width':
                self.config['protocolLineWidth'] = data
                self.settings_changed.emit('protocolLineWidth', data)
            
            # Handle file chooser actions
            elif childName == 'Bird List.Common Bird List.Choose File':
                self._choose_bird_list_file('Common Bird List', 'BirdListShort', 'shortbl')
            elif childName == 'Bird List.Full Bird List.Choose File':
                self._choose_bird_list_file('Full Bird List', 'BirdListLong', 'longbl')
            elif childName == 'Bird List.Known Calls List.Choose File':
                self._choose_bird_list_file('Known Calls List', 'KnownCallsList', 'knownCalls')
            elif childName == 'Bird List.Bat List.Choose File':
                self._choose_bird_list_file('Bat List', 'BatList', 'batl')
            elif childName == 'Bird List.Freebird List.Choose File':
                self._choose_freebird_list_file()

        # Reload the file to apply settings
        if self.parent_window and hasattr(self.parent_window, 'resetStorageArrays'):
            self.parent_window.resetStorageArrays()
            if hasattr(self.parent_window, 'loadFile') and hasattr(self.parent_window, 'session_filename'):
                self.parent_window.loadFile(self.parent_window.session_filename)

    def _choose_bird_list_file(self, list_name, config_key, loader_method):
        """Handle bird list file selection."""
        if self.parent_window:
            sound_file_dir = getattr(self.parent_window, 'SoundFileDir', self.configdir)
        else:
            sound_file_dir = self.configdir
            
        filename, _ = QFileDialog.getOpenFileName(
            self.parameter_widget, f'Choose {list_name}', sound_file_dir, "Text files (*.txt)"
        )
        
        if filename:
            # Load the list using the appropriate loader method
            loader_func = getattr(self.ConfigLoader, loader_method)
            new_list = loader_func(filename, self.configdir)
            
            if new_list is not None:
                self.config[config_key] = filename
                # Update the parameter tree display
                path_parts = list_name.split(' ')
                self.parameter_tree['Bird List', *path_parts, 'Filename'] = os.path.basename(filename)
                
                # Update the appropriate list
                if loader_method == 'shortbl':
                    self.shortBirdList = new_list
                elif loader_method == 'longbl':
                    self.longBirdList = new_list
                elif loader_method == 'knownCalls':
                    self.knownCalls = new_list
                    # Update with filter calls
                    for filt in self.FilterDicts.values():
                        if not filt["species"] in self.knownCalls:
                            self.knownCalls[filt["species"]] = []
                        for subf in filt["Filters"]:
                            if (not subf["calltype"] in self.knownCalls[filt["species"]] and 
                                not subf["calltype"] == "Not Specified" and 
                                not subf["calltype"] == "Add"):
                                self.knownCalls[filt["species"]].append(subf["calltype"])
                elif loader_method == 'batl':
                    self.batList = new_list
                    
                self.settings_changed.emit(config_key, filename)
            else:
                # Reload the original list
                original_list = loader_func(self.config[config_key], self.configdir)
                if loader_method == 'shortbl':
                    self.shortBirdList = original_list
                elif loader_method == 'longbl':
                    self.longBirdList = original_list
                elif loader_method == 'knownCalls':
                    self.knownCalls = original_list
                elif loader_method == 'batl':
                    self.batList = original_list

    def _choose_freebird_list_file(self):
        """Handle Freebird list file selection."""
        filename, _ = QFileDialog.getOpenFileName(
            self.parameter_widget, 'Choose Freebird List', self.configdir, "*.csv *.xlsx"
        )
        
        if filename:
            self.config['FreebirdList'] = filename
            self.parameter_tree['Bird List', 'Freebird List', 'Filename'] = os.path.basename(filename)
            self.settings_changed.emit('FreebirdList', filename)
    
    def reload_filters(self):
        """Reload filters from directory."""
        self.FilterDicts = self.ConfigLoader.filters(self.filtersDir)
        
    def save_config_to_file(self):
        """Save configuration to file."""
        self.ConfigLoader.configwrite(self.config, self.configfile)
        self.config_saved.emit()
    
    def save_bird_lists(self):
        """Save all bird lists to their respective files."""
        # Save each list to its configured file
        self.ConfigLoader.blwrite(self.longBirdList, self.config['BirdListLong'], self.configdir)
        self.ConfigLoader.blwrite(self.shortBirdList, self.config['BirdListShort'], self.configdir) 
        self.ConfigLoader.blwrite(self.batList, self.config['BatList'], self.configdir)
        if self.knownCalls:
            self.ConfigLoader.knownCallsWrite(self.knownCalls, self.config['KnownCallsList'], self.configdir)
    
    # Batch-specific configuration methods
    def get_batch_settings(self):
        """Return all batch-specific configuration settings."""
        return {
            'protocolSize': self.config.get('protocolSize', 15),
            'protocolInterval': self.config.get('protocolInterval', 600),
            'FiltersDir': self.config.get('FiltersDir', 'Filters'),
            'window_width': self.config.get('window_width', 512),
            'incr': self.config.get('incr', 128),
            'ColourNone': self.config.get('ColourNone', [200, 200, 200, 255]),
            'ColourPossible': self.config.get('ColourPossible', [255, 255, 0, 255]),
            'ColourNamed': self.config.get('ColourNamed', [0, 255, 0, 255])
        }
    
    def load_filters(self, filters_dir=None):
        """Load filter dictionaries from the specified directory.
        
        Args:
            filters_dir: Optional directory path. If None, uses config['FiltersDir']
            
        Returns:
            dict: Filter dictionaries loaded from the directory
        """
        if filters_dir is None:
            filters_dir = os.path.join(self.configdir, self.config['FiltersDir'])
        
        filter_dicts = self.ConfigLoader.filters(filters_dir)
        
        # Remove NZ Bats from the main filter list as it's handled specially
        if "NZ Bats" in filter_dicts:
            del filter_dicts["NZ Bats"]
            
        return filter_dicts
    
    def save_review_preferences(self, preferences):
        """Save review-specific UI preferences to config.
        
        Args:
            preferences: dict of preference settings to save
        """
        # Update config with provided preferences
        for key, value in preferences.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                # Emit signal if value changed
                if old_value != value:
                    self.settings_changed.emit(key, value)
        
        # Save to file
        self.save_config_to_file()
    
    def update_batch_processing_settings(self, protocol_size=None, protocol_interval=None):
        """Update batch processing-specific settings.
        
        Args:
            protocol_size: Size of processing window in seconds
            protocol_interval: Interval between processing windows in seconds
        """
        if protocol_size is not None:
            self.config['protocolSize'] = protocol_size
            self.settings_changed.emit('protocolSize', protocol_size)
            
        if protocol_interval is not None:
            self.config['protocolInterval'] = protocol_interval
            self.settings_changed.emit('protocolInterval', protocol_interval)
        
        self.save_config_to_file()
