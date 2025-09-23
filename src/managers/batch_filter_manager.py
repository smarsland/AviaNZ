# batch_filter_manager.py
# Part of AviaNZ refactoring - handles batch-specific filter and species selection logic

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox
import os
import json


class BatchFilterManager(QObject):
    """Manages batch-specific filter and species selection logic.
    
    Handles:
    - Filter loading and validation (leverages ConfigManager)
    - "Any sound" vs specific species coordination
    - NZ Bats special mode handling
    - Species selection validation for batch jobs
    """
    
    # Signals for filter operations
    filters_loaded = pyqtSignal(dict)  # available filters
    filter_selection_changed = pyqtSignal(list)  # selected filters
    species_selection_validated = pyqtSignal(bool)  # validation result
    special_mode_activated = pyqtSignal(str)  # mode name
    
    def __init__(self, config_manager, species_manager):
        super().__init__()
        self.config_manager = config_manager
        self.species_manager = species_manager
        
        # Filter state
        self.available_filters = {}
        self.selected_filters = []
        self.current_mode = "normal"  # normal, nz_bats, any_sound
        
    def load_available_filters(self):
        """Load available filters using ConfigManager.
        
        Returns:
            dict: Dictionary of available filters
        """
        self.available_filters = self.config_manager.load_filters()
        
        # Validate filter integrity
        valid_filters = {}
        for filter_name, filter_data in self.available_filters.items():
            if self._validate_filter_data(filter_data):
                valid_filters[filter_name] = filter_data
                
        self.available_filters = valid_filters
        self.filters_loaded.emit(self.available_filters)
        return self.available_filters
        
    def _validate_filter_data(self, filter_data):
        """Validate filter data structure.
        
        Args:
            filter_data: Filter configuration dictionary
            
        Returns:
            bool: True if filter data is valid
        """
        if not isinstance(filter_data, dict):
            return False
            
        # Check for required fields
        required_fields = ['species', 'SampleRate', 'windowing']
        for field in required_fields:
            if field not in filter_data:
                return False
                
        return True
        
    def validate_filter_selection(self, selected_filters):
        """Check user filter selections for validity.
        
        Args:
            selected_filters: List of selected filter names
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not selected_filters:
            return (False, "No filters selected")
            
        # Check all selected filters exist
        for filter_name in selected_filters:
            if filter_name not in self.available_filters:
                return (False, f"Filter '{filter_name}' not found")
                
        # Check for conflicting sample rates
        sample_rates = set()
        for filter_name in selected_filters:
            filter_data = self.available_filters[filter_name]
            sample_rates.add(filter_data.get('SampleRate', 0))
            
        if len(sample_rates) > 1:
            return (False, "Selected filters have conflicting sample rates")
            
        self.selected_filters = selected_filters
        self.filter_selection_changed.emit(selected_filters)
        return (True, "")
        
    def handle_special_modes(self, mode_name):
        """Handle NZ Bats and "Any sound" special modes.
        
        Args:
            mode_name: "nz_bats", "any_sound", or "normal"
        """
        self.current_mode = mode_name
        
        if mode_name == "nz_bats":
            # Load NZ bat-specific filters
            bat_filters = self._get_bat_filters()
            if bat_filters:
                self.selected_filters = bat_filters
                self.filter_selection_changed.emit(bat_filters)
                
        elif mode_name == "any_sound":
            # Clear specific filters for generic detection
            self.selected_filters = []
            self.filter_selection_changed.emit([])
            
        self.special_mode_activated.emit(mode_name)
        
    def _get_bat_filters(self):
        """Get filters suitable for bat detection.
        
        Returns:
            list: List of bat-specific filter names
        """
        bat_filters = []
        for filter_name, filter_data in self.available_filters.items():
            species = filter_data.get('species', '').lower()
            if 'bat' in species or filter_name.lower().startswith('bat'):
                bat_filters.append(filter_name)
                
        return bat_filters
        
    def get_batch_filter_config(self):
        """Return filter configuration for batch worker.
        
        Returns:
            dict: Configuration for batch processing
        """
        config = {
            'selected_filters': self.selected_filters,
            'mode': self.current_mode,
            'filter_data': {}
        }
        
        # Include full filter data for selected filters
        for filter_name in self.selected_filters:
            if filter_name in self.available_filters:
                config['filter_data'][filter_name] = self.available_filters[filter_name]
                
        return config
        
    def validate_species_for_batch(self, selected_species):
        """Validate species selection for batch processing.
        
        Args:
            selected_species: List of species names
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not selected_species and self.current_mode == "normal":
            return (False, "No species selected for batch processing")
            
        # Validate each species name
        for species in selected_species:
            is_valid, error_msg = self.species_manager.validate_species_name(species)
            if not is_valid:
                return (False, f"Invalid species '{species}': {error_msg}")
                
        # Check species compatibility with selected filters
        if self.selected_filters and selected_species:
            compatible = self._check_species_filter_compatibility(selected_species)
            if not compatible:
                return (False, "Selected species not compatible with chosen filters")
                
        self.species_selection_validated.emit(True)
        return (True, "")
        
    def _check_species_filter_compatibility(self, species_list):
        """Check if species are compatible with selected filters.
        
        Args:
            species_list: List of species names
            
        Returns:
            bool: True if compatible
        """
        for filter_name in self.selected_filters:
            filter_data = self.available_filters[filter_name]
            filter_species = filter_data.get('species', '').lower()
            
            # Check if any selected species matches filter
            for species in species_list:
                if species.lower() in filter_species or filter_species in species.lower():
                    return True
                    
        # If no exact matches, allow anyway (user may know better)
        return True
        
    def get_filter_info(self, filter_name):
        """Get detailed information about a specific filter.
        
        Args:
            filter_name: Name of the filter
            
        Returns:
            dict: Filter information or None if not found
        """
        return self.available_filters.get(filter_name)
        
    def get_selected_filters_info(self):
        """Get information about currently selected filters.
        
        Returns:
            dict: Information about selected filters
        """
        info = {
            'count': len(self.selected_filters),
            'filters': {},
            'species': set(),
            'sample_rates': set()
        }
        
        for filter_name in self.selected_filters:
            if filter_name in self.available_filters:
                filter_data = self.available_filters[filter_name]
                info['filters'][filter_name] = filter_data
                info['species'].add(filter_data.get('species', 'Unknown'))
                info['sample_rates'].add(filter_data.get('SampleRate', 0))
                
        # Convert sets to lists for JSON serialization
        info['species'] = list(info['species'])
        info['sample_rates'] = list(info['sample_rates'])
        
        return info
        
    def reset_selection(self):
        """Reset filter and mode selection to defaults."""
        self.selected_filters = []
        self.current_mode = "normal"
        self.filter_selection_changed.emit([])
        
    def export_filter_config(self, filepath):
        """Export current filter configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        config = self.get_batch_filter_config()
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
    def import_filter_config(self, filepath):
        """Import filter configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            bool: True if import successful
        """
        if not os.path.exists(filepath):
            return False
            
        with open(filepath, 'r') as f:
            config = json.load(f)
            
        # Validate imported config
        if 'selected_filters' in config:
            is_valid, _ = self.validate_filter_selection(config['selected_filters'])
            if is_valid:
                self.current_mode = config.get('mode', 'normal')
                return True
                
        return False