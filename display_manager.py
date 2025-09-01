# Display Manager for AviaNZ Manual
# Handles graphics rendering, spectrogram display, and visual elements

from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
import pyqtgraph as pg
import SupportClasses_GUI
import colourMaps

class DisplayManager(QObject):
    """Manages display rendering, spectrogram visualization, and graphics updates"""
    
    # Signals for UI coordination
    spectrogram_ready = pyqtSignal()
    overview_updated = pyqtSignal()
    graphics_refreshed = pyqtSignal()
    
    def __init__(self, config_manager, audio_processor=None):
        super().__init__()
        self.config_manager = config_manager
        self.config = config_manager.config
        self.audio_processor = audio_processor
        
        # Display state
        self.sg = None
        self.sgMinimum = 0
        self.sgMaximum = 1
        self.noisefloor = 0
        
        # Graphics references (set by main window)
        self.p_spec = None
        self.p_overview = None
        self.p_overview2 = None
        self.specPlot = None
        self.overviewImage = None
        self.overviewImageRegion = None
        self.amplPlot = None
        self.timeaxis = None
        self.specaxis = None
        self.guidelines = []
        
        # Labels for zooniverse mode
        self.label1 = None
        self.label2 = None
        self.label3 = None
        self.label4 = None
        self.label5 = None
        
        # Segment display state
        self.SegmentRects = []
        self.widthOverviewSegment = 0
        self.overviewSegments = None
        self.textpos = 0
        
        # Display properties
        self.batmode = False
        self.zooniverse = False
        
    def set_graphics_references(self, graphics_refs):
        """Set references to main window graphics objects"""
        self.p_spec = graphics_refs.get('p_spec')
        self.p_overview = graphics_refs.get('p_overview')
        self.p_overview2 = graphics_refs.get('p_overview2')
        self.specPlot = graphics_refs.get('specPlot')
        self.overviewImage = graphics_refs.get('overviewImage')
        self.overviewImageRegion = graphics_refs.get('overviewImageRegion')
        self.amplPlot = graphics_refs.get('amplPlot')
        self.timeaxis = graphics_refs.get('timeaxis')
        self.specaxis = graphics_refs.get('specaxis')
        self.guidelines = graphics_refs.get('guidelines', [])
        
    def set_display_mode(self, batmode=False, zooniverse=False):
        """Set display mode flags"""
        self.batmode = batmode
        self.zooniverse = zooniverse
        
    def set_noisefloor(self, noisefloor):
        """Set noise floor level"""
        self.noisefloor = noisefloor
        
    def render_main_displays(self, sp, datalength, datalengthSec, widthWindow, remaking=False):
        """Renders the main amplitude and spectrogram plots
        This is the core business logic from drawfigMain"""
        
        if len(sp.data) > 0 and not self.batmode and self.amplPlot:
            self.amplPlot.setData(np.linspace(0.0, datalengthSec, num=datalength, endpoint=True), sp.data)

        if self.timeaxis:
            self.timeaxis.setLabel('')

        self.update_figure_data(sp)
        
        # Only proceed with rendering if we have spectrogram data
        if self.sg is None:
            return 0  # Return default textpos
        
        # Sort out the spectrogram frequency axis
        FreqRange = sp.maxFreqShow - sp.minFreqShow
        height = sp.audioFormat.sampleRate() // 2 / np.shape(self.sg)[1]
        SpecRange = FreqRange / height
        
        self.render_frequency_guides()
        
        if self.zooniverse:
            self._render_zooniverse_labels(FreqRange, SpecRange, sp)
        else:
            self._render_standard_labels(FreqRange, SpecRange, sp)
            
        self.textpos = int((sp.maxFreqShow - sp.minFreqShow) / height)
        
        self.graphics_refreshed.emit()
        
        # Return textpos for main window
        return self.textpos
        
    def _render_zooniverse_labels(self, FreqRange, SpecRange, sp):
        """Render labels for zooniverse mode"""
        if not self.p_spec:
            return
            
        labels = [0, int(FreqRange//4000), int(FreqRange//2000), int(3*FreqRange//4000), int(FreqRange//1000)]
        
        if self.config['sgScale'] == 'Mel Frequency':
            for i in range(len(labels)):
                labels[i] = sp.convertHztoMel(labels[i])
        elif self.config['sgScale'] == 'Bark Frequency':
            for i in range(len(labels)):
                labels[i] = sp.convertHztoBark(labels[i])
        
        offset = 6
        positions = [0, SpecRange/4, SpecRange/2, 3*SpecRange/4, SpecRange]
        
        # Clear existing labels
        for label in [self.label1, self.label2, self.label3, self.label4, self.label5]:
            if label:
                self.p_spec.removeItem(label)
        
        # Create new labels
        label_items = []
        for i, (label_val, pos) in enumerate(zip(labels, positions)):
            txt = f'<span style="color: #0F0; font-size:20pt">{label_val}</div>'
            label_item = pg.TextItem(html=txt, color='g', anchor=(0,0))
            self.p_spec.addItem(label_item)
            label_item.setPos(0, pos + offset)
            label_items.append(label_item)
            
        self.label1, self.label2, self.label3, self.label4, self.label5 = label_items
        
    def _render_standard_labels(self, FreqRange, SpecRange, sp):
        """Render labels for standard mode"""
        if not self.specaxis:
            return
            
        labels = [
            sp.minFreqShow,
            sp.minFreqShow + FreqRange/4,
            sp.minFreqShow + FreqRange/2,
            sp.minFreqShow + 3*FreqRange/4,
            sp.minFreqShow + FreqRange
        ]

        if self.config['sgScale'] == 'Mel Frequency':
            for i in range(len(labels)):
                labels[i] = sp.convertHztoMel(labels[i])
            self.specaxis.setLabel('Mels')
        elif self.config['sgScale'] == 'Bark Frequency':
            for i in range(len(labels)):
                labels[i] = sp.convertHztoBark(labels[i]) * 1000
            self.specaxis.setLabel('Barks')
        else:
            self.specaxis.setLabel('kHz')
       
        ticks = [
            (0, round(labels[0]/1000, 2)),
            (SpecRange/4, round(labels[1]/1000, 2)),
            (SpecRange/2, round(labels[2]/1000, 2)),
            (3*SpecRange/4, round(labels[3]/1000, 2)),
            (SpecRange, round(labels[4]/1000, 2))
        ]
        ticks = [[(tick[0], "%.1f" % tick[1]) for tick in ticks]]
        self.specaxis.setTicks(ticks)
        
    def render_overview(self, sg):
        """Renders the overview display with the complete spectrogram"""
        if not self.overviewImage or not self.overviewImageRegion:
            return
            
        self.overviewImage.setImage(sg)
        self.overviewImageRegion.setBounds([0, len(sg)])
        
        if self.audio_processor:
            # Set initial region based on current window width
            from PyQt6.QtWidgets import QApplication
            # Get current window width from main window if available
            try:
                main_window = QApplication.instance().activeWindow()
                if hasattr(main_window, 'widthWindow'):
                    width_spec = self.audio_processor.convertAmpltoSpec(main_window.widthWindow.value())
                    self.overviewImageRegion.setRegion([0, width_spec])
            except:
                # Fallback to default region
                self.overviewImageRegion.setRegion([0, min(1000, len(sg))])
        
        self.overview_updated.emit()
        
    def update_figure_data(self, sp):
        """Updates the figure data without changing the UI layout"""
        if not sp or not hasattr(sp, 'audioFormat') or self.sg is None:
            return
            
        height = sp.audioFormat.sampleRate() // 2 / np.shape(self.sg)[1]
        pixelstart = int(sp.minFreqShow / height)
        pixelend = int(sp.maxFreqShow / height)

        # Set image data first
        if self.overviewImage:
            self.overviewImage.setImage(self.sg[:, pixelstart:pixelend])
        if self.overviewImageRegion:
            self.overviewImageRegion.setBounds([0, len(self.sg)])
        if self.specPlot:
            self.specPlot.setImage(self.sg[:, pixelstart:pixelend])

        # Apply settings after image data is set
        self.apply_spectrogram_settings()
        
    def apply_spectrogram_settings(self):
        """Applies current colour map and brightness/contrast settings"""
        self.set_colour_map(self.config['cmap'])
        self.set_colour_levels()
        
    def set_colour_map(self, cmap):
        """Sets the colour map for spectrogram displays"""
        self.config['cmap'] = cmap
        lut = colourMaps.getLookupTable(self.config['cmap'])

        if self.specPlot:
            self.specPlot.setLookupTable(lut)
        if self.overviewImage:
            self.overviewImage.setLookupTable(lut)
            
    def set_colour_levels(self, brightness=None, contrast=None):
        """Sets brightness and contrast levels for spectrogram displays"""
        if brightness is None:
            brightness = self.config['brightness']
        if contrast is None:
            contrast = self.config['contrast']
            
        # Use the same algorithm as the original colourMaps.getColourRange function
        import colourMaps
        levels = colourMaps.getColourRange(self.sgMinimum, self.sgMaximum, brightness, contrast, self.config['invertColourMap'])
        
        if self.specPlot:
            self.specPlot.setLevels(levels)
        if self.overviewImage:
            self.overviewImage.setLevels(levels)
            
    def render_frequency_guides(self):
        """Renders frequency guide lines"""
        if not self.p_spec or not self.audio_processor:
            return
            
        if self.config['guidelinesOn'] == 'always' or (self.config['guidelinesOn'] == 'bat' and self.batmode):
            for gi in range(len(self.guidelines)):
                if gi < len(self.config['guidepos']):
                    self.guidelines[gi].setValue(self.audio_processor.convertFreqtoY(self.config['guidepos'][gi]))
                    pen_color = self.config['guidecol'][gi] if gi < len(self.config['guidecol']) else 'white'
                    self.guidelines[gi].setPen(color=pen_color, width=2)
                    self.p_spec.addItem(self.guidelines[gi], ignoreBounds=True)
        else:
            # Hide guidelines
            for g in self.guidelines:
                g.setValue(-1000)
                
    def render_protocol_marks(self, segments, sp):
        """Renders protocol marks on the spectrogram"""
        if not self.p_spec or not sp:
            return
            
        # Implementation would depend on specific protocol mark requirements
        # This is a placeholder for the protocol marking functionality
        pass
        
    def prepare_spectrogram(self, sp, noisefloor_percent=0):
        """Prepares and normalizes the spectrogram data"""
        if not sp:
            return
            
        # Normalize spectrogram
        if self.batmode:
            self.sg = sp.normalisedSpec("Batmode")
        else:
            self.sg = sp.normalisedSpec(self.config['sgNormMode'])

        self.sgMinimum = np.min(self.sg)
        self.sgMaximum = np.max(self.sg)
        
        # Apply noise floor
        noisefloor = noisefloor_percent / 100 * (self.sgMaximum - self.sgMinimum) + self.sgMinimum
        self.sg = np.where(self.sg < noisefloor, 0, self.sg)
        
        # Update sgMinimum and sgMaximum after noise floor application
        self.sgMinimum = np.min(self.sg)
        self.sgMaximum = np.max(self.sg)
        
        # Apply initial color levels with current brightness/contrast settings
        self.set_colour_levels()
        
        self.spectrogram_ready.emit()
        
    def create_overview_segments(self, sg_shape, widthOverviewSegment):
        """Creates overview segment rectangles"""
        if not self.p_overview2:
            return
            
        # Calculate segments
        if self.audio_processor:
            numSegments = int(np.ceil(sg_shape[0] / self.audio_processor.convertAmpltoSpec(widthOverviewSegment)))
            self.widthOverviewSegment = sg_shape[0] // numSegments
        else:
            numSegments = 10  # Fallback
            self.widthOverviewSegment = sg_shape[0] // numSegments

        self.overviewSegments = np.zeros((numSegments, 3))

        # Delete existing overview segments
        for r in self.SegmentRects:
            self.p_overview2.removeItem(r)
        self.SegmentRects = []

        # Add new overview segments
        for i in range(numSegments):
            r = SupportClasses_GUI.ClickableRectItem(
                i * self.widthOverviewSegment, 0, 
                self.widthOverviewSegment, 1
            )
            r.setPen(pg.mkPen(100, 100, 100))
            r.setBrush(pg.mkBrush('w'))
            self.SegmentRects.append(r)
            self.p_overview2.addItem(r)
            
        if self.p_overview2:
            self.p_overview2.setYRange(-0.2, 1, padding=0.02)

    def setup_file_display(self, filename, datalength_sec, sp, start_time, start_read, zooniverse, config, batmode):
        """Set up display elements for a newly loaded file.
        
        Args:
            filename: Name of the loaded file
            datalength_sec: Duration of data in seconds  
            sp: Spectrogram processor instance
            start_time: Start time offset
            start_read: Read start position
            zooniverse: Whether in zooniverse mode
            config: Configuration dictionary
            batmode: Whether in bat mode
            
        Returns:
            None: Sets up timeaxis and nFileSections attributes
        """
        import re
        
        self.datalength_sec = datalength_sec
        self.start_time = start_time
        self.start_read = start_read
        
        # Create time axis based on file type
        if hasattr(self, 'timeaxis') and not zooniverse:
            # Remove existing timeaxis from main window's w_spec
            # Note: This removal needs to be handled by the main window
            pass
            
        # Check if filename is in DOC format for time axis type
        DOCRecording = re.search(r'(\d{6})_(\d{6})', filename[-17:-4])
        
        if DOCRecording:
            self.start_time = DOCRecording.group(2)
            
            if int(self.start_time[:2]) > 17 or int(self.start_time[:2]) < 7:  # 6pm to 6am
                print("Night time DOC recording")
            else:
                print("Day time DOC recording")
                
            self.start_time = int(self.start_time[:2]) * 3600 + int(self.start_time[2:4]) * 60 + int(self.start_time[4:6])
            # Note: p_ampl reference needs to be passed from main window
            self.timeaxis = SupportClasses_GUI.TimeAxisHour(orientation='bottom', linkView=None)
        else:
            self.start_time = 0
            self.timeaxis = SupportClasses_GUI.TimeAxisMin(orientation='bottom', linkView=None)
        
        # Calculate file sections
        if datalength_sec != sp.fileLength and not batmode:
            self.nFileSections = int(np.ceil(sp.fileLength / datalength_sec))
        else:
            self.nFileSections = 1
