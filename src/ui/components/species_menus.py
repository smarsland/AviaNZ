# coding=latin-1

# species_menus.py
# Species selection components for the AviaNZ program


# Version 3.4 18/12/24
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

import copy
from functools import partial
from PyQt6.QtWidgets import QMenu
from PyQt6.QtCore import pyqtSignal


class BaseSpeciesMenu(QMenu):
    """Base class for species selection menus with common functionality."""
    
    # Common signals - using camelCase to match existing code conventions
    labelsUpdated = pyqtSignal(object, str, object, int)  # labels, species, call_type, certainty
    addSpecies = pyqtSignal(int)  # certainty
    addCallname = pyqtSignal(str, int)  # species, certainty
    
    def __init__(self, current_labels, parent=None, unsure=False, multiple_birds=False):
        super().__init__(parent)
        self.current_labels = copy.deepcopy(current_labels) if current_labels else []
        self.unsure = unsure
        self.multiple_birds = multiple_birds
        self.parent = parent
        
    def _format_label(self, label):
        """Add uncertainty marker if needed."""
        return f"{label}?" if self.unsure and label != "Don't Know" else label
    
    def _parse_species_name(self, species_text):
        """Parse species name from various formats."""
        if '>' in species_text:
            parts = species_text.split('>')
            return parts[-1].strip()
        return species_text
    
    def _get_certainty(self, species):
        """Get certainty level based on species and unsure state."""
        if species == "Don't Know":
            return 0
        elif self.unsure:
            return 50
        else:
            return 100
    
    def _get_current_species_list(self):
        """Get list of currently selected species names."""
        return [label.get('species', '') for label in self.current_labels]
    
    def _emit_labels_updated(self, species, call_type, certainty):
        """Emit the labels updated signal."""
        self.labelsUpdated.emit(
            copy.deepcopy(self.current_labels), 
            species, 
            call_type, 
            certainty
        )
    
    def refresh_species_menu(self, species_name):
        """Refresh a specific species menu to show updated call types."""
        # Find the species menu by title and rebuild it
        for action in self.actions():
            if hasattr(action, 'menu') and action.menu() and action.menu().title() == species_name:
                self._rebuild_species_submenu(action.menu(), species_name)
                break
    
    def _rebuild_species_submenu(self, species_menu, species_name):
        """Rebuild a species submenu with updated call types - to be implemented by subclasses."""
        pass


class BirdSelectionMenu(BaseSpeciesMenu):
    """Enhanced bird species selection menu with hierarchical organization."""
    
    def __init__(self, shortBirdList, longBirdList, knownCalls, currentLabels, 
                 parent=None, unsure=False, multipleBirds=False, includeCalltype=True):
        super().__init__(currentLabels, parent, unsure, multipleBirds)
        self.include_call_type = includeCalltype
        self.known_calls = knownCalls or {}
        
        self._create_short_menu(shortBirdList)
        self._create_long_menu(longBirdList)
    
    def _create_short_menu(self, short_bird_list):
        """Create the main menu from short bird list."""
        if not short_bird_list:
            return
            
        current_species = self._get_current_species_list()
        
        for item in short_bird_list:
            species = self._parse_species_name(item)
            
            if species == "Don't Know":
                self._add_dont_know_action(current_species)
            elif not self.include_call_type:
                self._add_species_simple(species, current_species)
            else:
                self._add_species_with_calls(species, current_species)
    
    def _add_dont_know_action(self, current_species):
        """Add Don't Know action."""
        label = "\u2714 Don't Know" if "Don't Know" in current_species else "Don't Know"
        action = self.addAction(label)
        action.triggered.connect(partial(self._species_selected, "Don't Know", None))
    
    def _add_species_simple(self, species, current_species):
        """Add species without call type submenu."""
        label = self._format_label(species)
        if species in current_species:
            label = f"\u2714 {label}"
        action = self.addAction(label)
        action.triggered.connect(partial(self._species_selected, species, None))
    
    def _add_species_with_calls(self, species, current_species):
        """Add species with call type submenu."""
        call_types = self.known_calls.get(species, [])
        call_types = ["Not Specified"] + call_types + ["Add"]
        
        # Create submenu for this species
        species_menu = QMenu(species, self)
        self.addMenu(species_menu)
        
        # Check if any calls for this species are currently selected
        any_checked = False
        current_call_type = None
        
        if species in current_species:
            # Find the current call type for this species
            for label in self.current_labels:
                if label.get('species') == species:
                    current_call_type = label.get('call_type', 'Not Specified')
                    any_checked = True
                    break
        
        # Add call type actions
        for call_type in call_types:
            if call_type == "Add":
                action = species_menu.addAction("Add new call type...")
                action.triggered.connect(partial(self._add_call_type_requested, species))
            else:
                call_label = self._format_label(call_type)
                if call_type == current_call_type:
                    call_label = f"\u2714 {call_label}"
                action = species_menu.addAction(call_label)
                action.triggered.connect(partial(self._species_selected, species, call_type))
        
        # Update species menu title if any calls are selected
        if any_checked:
            species_menu.setTitle(f"\u2714 {species}")
        else:
            species_menu.setTitle(species)
    
    def _rebuild_species_submenu(self, species_menu, species_name):
        """Rebuild a species submenu with updated call types."""
        # Get current species selections
        current_species = [label['species'] for label in self.current_labels]
        
        # Rebuild the menu using the same logic as _add_species_with_calls
        call_types = self.known_calls.get(species_name, [])
        call_types = ["Not Specified"] + call_types + ["Add"]
        
        # Check if any calls for this species are currently selected
        any_checked = False
        current_call_type = None
        
        if species_name in current_species:
            # Find the current call type for this species
            for label in self.current_labels:
                if label.get('species') == species_name:
                    current_call_type = label.get('call_type', 'Not Specified')
                    any_checked = True
                    break
        
        # Add call type actions
        for call_type in call_types:
            if call_type == "Add":
                action = species_menu.addAction("Add new call type...")
                action.triggered.connect(partial(self._add_call_type_requested, species_name))
            else:
                call_label = self._format_label(call_type)
                if call_type == current_call_type:
                    call_label = f"\u2714 {call_label}"
                action = species_menu.addAction(call_label)
                action.triggered.connect(partial(self._species_selected, species_name, call_type))
        
        # Update species menu title if any calls are selected
        if any_checked:
            species_menu.setTitle(f"\u2714 {species_name}")
        else:
            species_menu.setTitle(species_name)
    
    def _create_long_menu(self, long_bird_list):
        """Create 'See all' hierarchical menu."""
        if not long_bird_list:
            return
            
        see_all_menu = QMenu("See all", self)
        self.addMenu(see_all_menu)
        
        # Build hierarchical tree: Letter -> Genus -> Species -> Calls
        bird_tree = self._build_bird_tree(long_bird_list)
        
        # Create menu structure
        for letter in sorted(bird_tree.keys()):
            letter_menu = QMenu(letter, see_all_menu)
            see_all_menu.addMenu(letter_menu)
            self._add_genus_to_menu(letter_menu, bird_tree[letter])
        
        # Add "Add new species" option
        add_action = see_all_menu.addAction("Add new species...")
        add_action.triggered.connect(partial(self._add_species_requested))
    
    def _build_bird_tree(self, long_bird_list):
        """Build hierarchical tree structure from bird list."""
        tree = {}
        
        for entry in long_bird_list:
            species = self._parse_species_name(entry)
            if not species or species == "Don't Know":
                continue
                
            # Get first letter
            first_letter = species[0].upper()
            if first_letter not in tree:
                tree[first_letter] = {}
            
            # Extract genus (first word)
            words = species.split()
            if len(words) > 1:
                genus = words[0]
                species_name = ' '.join(words[1:])
                
                if genus not in tree[first_letter]:
                    tree[first_letter][genus] = {}
                
                tree[first_letter][genus][species_name] = species
            else:
                # Single word species - put directly under letter
                tree[first_letter][None] = tree[first_letter].get(None, {})
                tree[first_letter][None][species] = species
                
        return tree
    
    def _add_genus_to_menu(self, parent_menu, genus_dict):
        """Add genus and its species to menu."""
        if None in genus_dict:
            # Direct species under letter
            for species_name, full_name in genus_dict[None].items():
                self._add_species_action_to_menu(parent_menu, full_name)
            del genus_dict[None]
        
        for genus, species_dict in genus_dict.items():
            if len(species_dict) == 1:
                # Only one species in genus - add directly
                full_name = list(species_dict.values())[0]
                self._add_species_action_to_menu(parent_menu, full_name)
            else:
                # Multiple species - create genus submenu
                genus_menu = QMenu(genus, parent_menu)
                parent_menu.addMenu(genus_menu)
                for species_name, full_name in species_dict.items():
                    self._add_species_action_to_menu(genus_menu, full_name)
    
    def _add_species_action_to_menu(self, parent_menu, species):
        """Add a single species action to a menu."""
        current_species = self._get_current_species_list()
        
        if not self.include_call_type:
            # Simple species action
            label = self._format_label(species)
            if species in current_species:
                label = f"\u2714 {label}"
            action = parent_menu.addAction(label)
            action.triggered.connect(partial(self._species_selected, species, None))
        else:
            # Species with call types - create submenu
            self._add_species_with_calls_to_menu(parent_menu, species, current_species)
    
    def _add_species_with_calls_to_menu(self, parent_menu, species, current_species):
        """Add species with call type submenu to a menu."""
        call_types = self.known_calls.get(species, [])
        call_types = ["Not Specified"] + call_types + ["Add"]
        
        # Create submenu for this species
        species_menu = QMenu(species, parent_menu)
        parent_menu.addMenu(species_menu)
        
        # Check if any calls for this species are currently selected
        any_checked = False
        current_call_type = None
        
        if species in current_species:
            # Find the current call type for this species
            for label in self.current_labels:
                if label.get('species') == species:
                    current_call_type = label.get('call_type', 'Not Specified')
                    any_checked = True
                    break
        
        # Add call type actions
        for call_type in call_types:
            if call_type == "Add":
                action = species_menu.addAction("Add new call type...")
                action.triggered.connect(partial(self._add_call_type_requested, species))
            else:
                call_label = self._format_label(call_type)
                if call_type == current_call_type:
                    call_label = f"\u2714 {call_label}"
                action = species_menu.addAction(call_label)
                action.triggered.connect(partial(self._species_selected, species, call_type))
        
        # Update species menu title if any calls are selected
        if any_checked:
            species_menu.setTitle(f"\u2714 {species}")
    
    def _species_selected(self, species, call_type):
        """Handle species selection."""
        if species == "Add":
            self._add_species_requested()
            return
        
        certainty = self._get_certainty(species)
        
        # Handle special unsure case
        if self.unsure and species.endswith('?'):
            species = species[:-1]
        
        self._update_labels(species, call_type, certainty)
        self._emit_labels_updated(species, call_type, certainty)
    
    def _add_species_requested(self):
        """Handle add new species request."""
        certainty = self._get_certainty("")
        self.addSpecies.emit(certainty)
    
    def _add_call_type_requested(self, species):
        """Handle add new call type request."""
        certainty = self._get_certainty(species)
        self.addCallname.emit(species, certainty)
    
    def _update_labels(self, species, call_type, certainty):
        """Update the current labels list."""
        current_species = self._get_current_species_list()
        
        if species == "Don't Know":
            self.current_labels = [{'species': "Don't Know", 'certainty': 0}]
            return
        
        # Remove existing entry for this species if present
        if species in current_species:
            self.current_labels = [
                label for label in self.current_labels 
                if label.get('species') != species
            ]
        
        # Clear existing labels if not multiple birds mode
        if not self.multiple_birds:
            self.current_labels.clear()
        
        # Add new label
        new_label = {'species': species, 'certainty': certainty}
        if call_type is not None:
            new_label['call_type'] = call_type
        
        self.current_labels.append(new_label)
        
        # Remove "Don't Know" if we have actual species
        self.current_labels = [
            label for label in self.current_labels 
            if label.get('species') != "Don't Know"
        ]
        
        # Ensure we have at least one label
        if not self.current_labels:
            self.current_labels = [{'species': "Don't Know", 'certainty': 0}]


class BatSelectionMenu(BaseSpeciesMenu):
    """Simplified bat species selection menu."""
    
    def __init__(self, batList, currentLabels, parent=None, unsure=False, multipleBirds=False):
        super().__init__(currentLabels, parent, unsure, multipleBirds)
        self._create_bat_menu(batList)
    
    def _create_bat_menu(self, bat_list):
        """Create bat selection menu."""
        if not bat_list:
            return
            
        current_species = self._get_current_species_list()
        
        for item in bat_list:
            species = self._parse_species_name(item)
            
            if species == "Don't Know":
                label = "\u2714 Don't Know" if "Don't Know" in current_species else "Don't Know"
            else:
                label = self._format_label(species)
                if species in current_species:
                    label = f"\u2714 {label}"
            
            action = self.addAction(label)
            action.triggered.connect(partial(self._bat_selected, species))
    
    def _bat_selected(self, species):
        """Handle bat selection."""
        certainty = self._get_certainty(species)
        
        # Handle special unsure case
        if self.unsure and species.endswith('?'):
            species = species[:-1]
        
        self._update_labels(species, certainty)
        self._emit_labels_updated(species, None, certainty)
    
    def _update_labels(self, species, certainty):
        """Update the current labels list for bats."""
        current_species = self._get_current_species_list()
        
        if species == "Don't Know":
            self.current_labels = [{'species': "Don't Know", 'certainty': 0}]
            return
        
        # Toggle species selection
        if species in current_species:
            # Remove the species
            self.current_labels = [
                label for label in self.current_labels 
                if label.get('species') != species
            ]
        else:
            # Add the species (clear others if not multiple birds mode)
            if not self.multiple_birds:
                self.current_labels.clear()
            self.current_labels.append({'species': species, 'certainty': certainty})
        
        # Remove "Don't Know" if we have actual species
        self.current_labels = [
            label for label in self.current_labels 
            if label.get('species') != "Don't Know"
        ]
        
        # Ensure we have at least one label
        if not self.current_labels:
            self.current_labels = [{'species': "Don't Know", 'certainty': 0}]