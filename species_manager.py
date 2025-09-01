# Species Manager - Handles species data operations and validation
# Version 3.4 18/12/24

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QInputDialog
import re
import copy


class SpeciesManager(QObject):
    """Manages species data operations, validation, and persistence.
    
    Handles all species-related operations including:
    - Species list management and reordering
    - Label validation and formatting
    - Data persistence through ConfigManager
    - Species and call type creation/validation
    """
    
    # Signals for species operations
    species_added = pyqtSignal(str, list)  # species_name, updated_calls
    call_type_added = pyqtSignal(str, str, list)  # species, call_type, updated_calls
    species_lists_updated = pyqtSignal()
    
    def __init__(self, config_manager):
        super().__init__()
        self.config_manager = config_manager
        self.config = config_manager.config
        
    def clean_text_encoding(self, text):
        """Clean up text encoding issues, particularly with apostrophes."""
        if not text:
            return text
            
        # Fix common encoding issues with apostrophes
        text = text.replace('â', "'")  # Fix corrupted apostrophe
        text = text.replace(''', "'")  # Fix smart quote to regular apostrophe
        text = text.replace(''', "'")  # Fix another smart quote variant
        
        return text
    
    def clean_species_list(self, species_list):
        """Clean a species list to fix encoding issues."""
        if not species_list:
            return species_list
            
        cleaned_list = []
        for species in species_list:
            if isinstance(species, str):
                cleaned_species = self.clean_text_encoding(species)
                cleaned_list.append(cleaned_species)
            else:
                cleaned_list.append(species)
                
        return cleaned_list
    
    def parse_species_name(self, name_string):
        """Parse species name from various formats.
        
        Handles formats like:
        - "Genus species"
        - "Genus (species)" 
        - "Genus>species"
        
        Returns: (display_name, storage_name, certainty)
        """
        if not name_string or name_string.strip() == "":
            return None, None, 0
            
        name = name_string.strip()
        
        # Handle uncertainty marker
        certainty = 100
        if name.endswith('?') and name != "Don't Know":
            name = name[:-1]
            certainty = 50
        elif name == "Don't Know":
            certainty = 0
            
        # Handle different name formats
        # Format: "Genus (species)"
        paren_match = re.match(r'^(.*?)\s*\((.*?)\)$', name)
        if paren_match:
            genus = paren_match.group(1).strip()
            species = paren_match.group(2).strip()
            display_name = f"{genus} ({species})"
            storage_name = f"{genus}>{species}"
            return display_name, storage_name, certainty
            
        # Format: "Genus>species" 
        if '>' in name:
            parts = name.split('>', 1)
            genus = parts[0].strip()
            species = parts[1].strip()
            display_name = f"{genus} ({species})"
            storage_name = name
            return display_name, storage_name, certainty
            
        # Simple format: "Genus species" or just "Genus"
        display_name = name
        storage_name = name
        return display_name, storage_name, certainty
    
    def validate_species_name(self, name):
        """Validate a species name for creation.
        
        Returns: (is_valid, error_message)
        """
        if not name or len(name.strip()) == 0:
            return False, "Species name cannot be empty"
            
        if len(name) > 150:
            return False, "Species name is too long (max 150 characters)"
            
        name = name.strip()
        
        # Check for reserved names
        reserved_names = ["don't know", "other", "(other)"]
        if name.lower() in reserved_names:
            return False, f"Name '{name}' is reserved and cannot be used"
            
        # Check for reserved characters
        if '?' in name:
            return False, "Species name cannot contain '?' character"
            
        # Validate format
        if not re.match(r'^[A-Za-z0-9\s\(\)\-\.]+$', name):
            return False, "Species name contains invalid characters"
            
        return True, ""
    
    def validate_call_type(self, call_type):
        """Validate a call type name for creation.
        
        Returns: (is_valid, error_message)
        """
        if not call_type or len(call_type.strip()) == 0:
            return False, "Call type cannot be empty"
            
        if len(call_type) > 150:
            return False, "Call type is too long (max 150 characters)"
            
        call_type = call_type.strip()
        
        # Check for reserved names
        reserved_names = ["don't know", "other", "(other)", "not specified"]
        if call_type.lower() in reserved_names:
            return False, f"Call type '{call_type}' is reserved and cannot be used"
            
        # Check for reserved characters
        if '?' in call_type:
            return False, "Call type cannot contain '?' character"
            
        return True, ""
    
    def reorder_short_list(self, short_list, segment_labels):
        """Reorder short bird list based on recent usage.
        
        Args:
            short_list: Current short bird list
            segment_labels: Labels from current segment
            
        Returns: Reordered list with recent species at top
        """
        if not self.config.get('ReorderList', True) or not segment_labels:
            return self._normalize_short_list(short_list)
            
        updated_list = short_list.copy()
        
        # Move segment species to front
        for label in segment_labels:
            species = label.get('species', '')
            if species and species in updated_list:
                updated_list.remove(species)
                updated_list.insert(0, species)
            elif species and len(updated_list) > 0:
                # Remove last item to make room
                updated_list.pop()
                updated_list.insert(0, species)
                
        return self._normalize_short_list(updated_list[:30])  # Limit to 30 items
    
    def _normalize_short_list(self, bird_list):
        """Normalize short bird list - move blanks to end, Don't Know to start, and clean encoding."""
        # Clean encoding issues first
        bird_list = self.clean_species_list(bird_list)
        
        # Remove empty strings and put them at the end
        non_empty = [x for x in bird_list if x.strip() != ""]
        empty = [x for x in bird_list if x.strip() == ""]
        
        # Move "Don't Know" to front
        dont_know = [x for x in non_empty if x == "Don't Know"]
        others = [x for x in non_empty if x != "Don't Know"]
        
        return dont_know + others + empty
    
    def create_species_labels(self, species_name, call_type=None, certainty=100, current_labels=None, multiple_birds=False):
        """Create or update species labels for a segment.
        
        Args:
            species_name: Name of the species
            call_type: Optional call type ("Not Specified" or None for none)
            certainty: Certainty level (0-100)
            current_labels: Existing labels list
            multiple_birds: Whether multiple species are allowed
            
        Returns: Updated labels list
        """
        if current_labels is None:
            current_labels = []
        else:
            current_labels = copy.deepcopy(current_labels)
            
        # Handle "Don't Know" special case
        if species_name == "Don't Know":
            return [{"species": "Don't Know", "certainty": 0}]
            
        # Process call type
        if call_type == "Not Specified":
            call_type = None
            
        # Check if species already exists
        existing_index = None
        for i, label in enumerate(current_labels):
            if label.get('species') == species_name:
                existing_index = i
                break
                
        # Create new label
        new_label = {"species": species_name, "certainty": certainty}
        if call_type:
            new_label["calltype"] = call_type
            
        if existing_index is not None:
            # Species exists - check if we're toggling or updating
            existing_label = current_labels[existing_index]
            existing_call = existing_label.get('calltype')
            
            if existing_call == call_type:
                # Same species and call type - remove (toggle off)
                current_labels.pop(existing_index)
            else:
                # Different call type - update
                current_labels[existing_index] = new_label
        else:
            # New species
            if not multiple_birds:
                # Single bird mode - replace all
                current_labels = [new_label]
            else:
                # Multiple birds mode - add to list
                current_labels.append(new_label)
                
        # Remove "Don't Know" if we have other species
        current_labels = [label for label in current_labels 
                         if not (label.get('species') == "Don't Know" and len(current_labels) > 1)]
        
        # If no labels left, add "Don't Know"
        if not current_labels:
            current_labels = [{"species": "Don't Know", "certainty": 0}]
            
        return current_labels
    
    def add_new_species(self, parent_widget, certainty=100):
        """Show dialog to add a new species.
        
        Returns: (success, species_name) tuple
        """
        species, ok = QInputDialog.getText(
            parent_widget, 
            'Bird name', 
            'Enter the bird name as genus (species)'
        )
        
        if not ok or not species:
            return False, None
            
        species = str(species).title().strip()
        
        # Validate the name
        is_valid, error_msg = self.validate_species_name(species)
        if not is_valid:
            print(f"ERROR: {error_msg}")
            return False, None
            
        display_name, storage_name, _ = self.parse_species_name(species)
        
        # Check if already exists
        long_list = self.config_manager.longBirdList
        if display_name in long_list or storage_name in long_list:
            print(f"Warning: Species '{display_name}' already exists")
            return False, None
            
        # Add to lists
        long_list.append(storage_name)
        long_list.sort(key=str.lower)
        
        # Initialize calls list
        known_calls = self.config_manager.knownCalls
        if display_name not in known_calls:
            known_calls[display_name] = []
            
        # Save to config
        self.config_manager.ConfigLoader.blwrite(
            long_list, 
            self.config['BirdListLong'], 
            self.config_manager.configdir
        )
        
        # Emit signal
        self.species_added.emit(display_name, known_calls.get(display_name, []))
        
        return True, display_name
    
    def add_new_call_type(self, parent_widget, species_name, certainty=100):
        """Show dialog to add a new call type for a species.
        
        Returns: (success, call_type) tuple
        """
        call_type, ok = QInputDialog.getText(
            parent_widget,
            'Call type',
            f'Enter a label for this call type for {species_name}'
        )
        
        if not ok or not call_type:
            return False, None
            
        call_type = str(call_type).title().strip()
        
        # Validate the call type
        is_valid, error_msg = self.validate_call_type(call_type)
        if not is_valid:
            print(f"ERROR: {error_msg}")
            return False, None
            
        # Check if already exists for this species
        known_calls = self.config_manager.knownCalls
        if species_name not in known_calls:
            known_calls[species_name] = []
            
        if call_type in known_calls[species_name]:
            print(f"Warning: Call type '{call_type}' already exists for {species_name}")
            return False, None
            
        # Add to calls list
        known_calls[species_name].append(call_type)
        
        # Emit signal  
        self.call_type_added.emit(species_name, call_type, known_calls[species_name])
        
        return True, call_type
    def update_segment_labels(self, current_labels, species_name, call_type=None, certainty=100, multiple_birds=False):
        """Update segment labels when a species is selected.
        
        This is the main business logic for handling species selection.
        Returns the updated labels and information about what changed.
        
        Returns: (updated_labels, reorder_info)
        where reorder_info = {"species": species_name, "should_reorder": bool}
        """
        updated_labels = self.create_species_labels(
            species_name=species_name,
            call_type=call_type, 
            certainty=certainty,
            current_labels=current_labels,
            multiple_birds=multiple_birds
        )
        
        reorder_info = {
            "species": species_name,
            "should_reorder": self.config.get('ReorderList', True) and species_name != "Don't Know"
        }
        
        return updated_labels, reorder_info
    
    def reorder_species_list(self, species_list, species_to_promote):
        """Reorder a species list by moving a species to the front.
        
        Args:
            species_list: List to reorder
            species_to_promote: Species name to move to front
            
        Returns: Reordered list
        """
        if not species_to_promote or species_to_promote == "Don't Know":
            return species_list
            
        updated_list = species_list.copy()
        
        if species_to_promote in updated_list:
            updated_list.remove(species_to_promote)
        elif len(updated_list) > 0:
            # Remove last item to make room
            updated_list.pop()
            
        updated_list.insert(0, species_to_promote)
        return updated_list
    
    def get_certainty_color_info(self, certainty):
        """Get color information based on certainty level.
        
        Returns: Color type string that main window can map to actual colors
        """
        if certainty == 0:
            return "none"  # ColourNone
        elif certainty == 50:
            return "possible"  # ColourPossibleDark  
        else:
            return "named"  # ColourNamed
    
    def handle_species_selection(self, current_labels, species_name, call_type=None, certainty=100, 
                                multiple_birds=False, species_lists=None):
        """Complete handler for species selection - all business logic in one place.
        
        Args:
            current_labels: Current segment labels
            species_name: Selected species
            call_type: Selected call type (or None)
            certainty: Certainty level
            multiple_birds: Whether multiple species are allowed
            species_lists: Dict with 'short_list' and 'bat_list' for reordering
            
        Returns: Dict with all information needed for UI updates:
        {
            'updated_labels': [...],
            'color_type': 'none'|'possible'|'named',
            'reorder_short_list': [...] or None,
            'reorder_bat_list': [...] or None,
            'last_species_info': {...} for next segment
        }
        """
        # Update labels
        updated_labels, reorder_info = self.update_segment_labels(
            current_labels, species_name, call_type, certainty, multiple_birds
        )
        
        # Determine color
        color_type = self.get_certainty_color_info(certainty)
        
        # Handle list reordering
        result = {
            'updated_labels': updated_labels,
            'color_type': color_type,
            'reorder_short_list': None,
            'reorder_bat_list': None,
            'last_species_info': None
        }
        
        if reorder_info['should_reorder'] and species_lists:
            if 'short_list' in species_lists:
                result['reorder_short_list'] = self.reorder_species_list(
                    species_lists['short_list'], species_name
                )
            if 'bat_list' in species_lists:
                result['reorder_bat_list'] = self.reorder_species_list(
                    species_lists['bat_list'], species_name
                )
        
        # Store info for next segment
        if species_name != "Don't Know":
            if call_type == "Not Specified" or call_type is None:
                result['last_species_info'] = {
                    "species": species_name, 
                    "certainty": 100, 
                    "filter": "M"
                }
            else:
                result['last_species_info'] = {
                    "species": species_name, 
                    "certainty": certainty, 
                    "filter": "M", 
                    "calltype": call_type
                }
        
        return result
