# batch_process_manager.py
# Part of AviaNZ refactoring - handles batch processing coordination and threading

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from ..ui import SupportClasses_GUI
from ..core.AviaNZ_batch import AviaNZ_batchProcess, GentleExitException
import traceback


class BatchProcessWorker(AviaNZ_batchProcess, QObject):
    """Worker class that adds QObject functionality to standard batchProc for threading."""
    
    # Signals for batch processing events
    finished = pyqtSignal()
    completed = pyqtSignal()
    stopped = pyqtSignal()
    failed = pyqtSignal(str)
    need_msg = pyqtSignal(str, str)
    need_clean_UI = pyqtSignal(int, int)
    need_update = pyqtSignal(int, str)
    need_bat_info = pyqtSignal(str, str, str, str)

    def __init__(self, *args, **kwargs):
        AviaNZ_batchProcess.__init__(self, *args, **kwargs)
        QObject.__init__(self)

    @pyqtSlot()
    def detect(self):
        """Main detection method run in worker thread."""
        try:
            AviaNZ_batchProcess.detect(self)
            self.completed.emit()
        except GentleExitException:
            # for clean exits, such as stops via progress dialog
            self.stopped.emit()
        except Exception as e:
            # we have UI, so just cleanly present the error;
            # in other modes this will CTD
            e = "Encountered error:\n" + traceback.format_exc()
            self.failed.emit(e)
        self.finished.emit()


class BatchProcessManager(QObject):
    """Manages batch processing workflows and thread coordination.
    
    Handles:
    - Batch worker thread coordination
    - Progress tracking and estimation
    - User cancellation handling
    - Parameter validation for batch jobs
    """
    
    # Signals for batch processing coordination
    processing_started = pyqtSignal()
    processing_completed = pyqtSignal()
    processing_stopped = pyqtSignal()
    processing_failed = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        
        # Thread management
        self.batch_thread = None
        self.batch_worker = None
        
        # Progress tracking
        self.progress_dialog = None
        self.msg_response = None
        
        # State
        self.is_processing = False
        
    def start_batch_processing(self, config_manager, dir_name, species, **batch_params):
        """Start batch processing workflow.
        
        Args:
            config_manager: ConfigManager instance
            dir_name: Directory to process
            species: Species to detect
            **batch_params: Additional batch processing parameters
        """
        if self.is_processing:
            return False
            
        self.is_processing = True
        
        # Validate parameters
        if not self.validate_batch_parameters(dir_name, species):
            self.is_processing = False
            return False
            
        # Create worker and thread
        self.batch_worker = BatchProcessWorker(
            self.parent_window, 
            mode="GUI", 
            configdir=config_manager.configdir, 
            sdir=dir_name, 
            recognisers=species,
            **batch_params
        )
        
        self.batch_thread = QThread()
        self.batch_worker.moveToThread(self.batch_thread)
        
        # Connect signals
        self._connect_worker_signals()
        
        # Start processing
        self.batch_thread.started.connect(self.batch_worker.detect)
        self.batch_thread.start()
        
        self.processing_started.emit()
        return True
        
    def _connect_worker_signals(self):
        """Connect worker signals to manager handlers."""
        self.batch_worker.finished.connect(self.batch_thread.quit)
        self.batch_worker.completed.connect(self._on_completed)
        self.batch_worker.stopped.connect(self._on_stopped)
        self.batch_worker.failed.connect(self._on_failed)
        self.batch_worker.need_msg.connect(self._handle_message_request)
        self.batch_worker.need_clean_UI.connect(self._handle_ui_cleanup)
        self.batch_worker.need_update.connect(self._handle_progress_update)
        self.batch_worker.need_bat_info.connect(self._handle_bat_survey_request)
        
    def _on_completed(self):
        """Handle successful completion."""
        self.is_processing = False
        self.processing_completed.emit()
        
    def _on_stopped(self):
        """Handle processing stop."""
        self.is_processing = False
        self.processing_stopped.emit()
        
    def _on_failed(self, error_message):
        """Handle processing failure."""
        self.is_processing = False
        self.processing_failed.emit(error_message)
        
    def _handle_message_request(self, title, text):
        """Handle message popup requests from worker."""
        msg = SupportClasses_GUI.MessagePopup("t", title, text)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        response = msg.exec()

        if response == QMessageBox.StandardButton.Cancel:
            self.msg_response = 2
        elif response == QMessageBox.StandardButton.No:
            self.msg_response = 1
        else:
            self.msg_response = 0
            
        # Signal back to worker
        if hasattr(self.batch_worker, 'msgClosed'):
            self.batch_worker.msgClosed.wakeAll()
            
    def _handle_ui_cleanup(self, cnt, total):
        """Handle UI cleanup requests."""
        # Progress dialog setup/cleanup can be handled here
        pass
        
    def _handle_progress_update(self, cnt, progress_text):
        """Handle progress updates from worker."""
        self.progress_updated.emit(cnt, progress_text)
        
    def _handle_bat_survey_request(self, *args):
        """Handle bat survey form requests."""
        # Forward to parent window if it has this capability
        if hasattr(self.parent_window, 'bat_survey_form'):
            self.parent_window.bat_survey_form(*args)
            
    def validate_batch_parameters(self, dir_name, species):
        """Validate batch processing parameters.
        
        Args:
            dir_name: Directory path to validate
            species: Species list to validate
            
        Returns:
            bool: True if parameters are valid
        """
        if not dir_name or dir_name == '':
            return False
            
        if not species:
            return False
            
        return True
        
    def manage_cancellation(self):
        """Handle user cancellation of batch processing."""
        if self.is_processing and self.batch_worker:
            # Signal worker to stop
            if hasattr(self.batch_worker, 'stop'):
                self.batch_worker.stop()
        
    def cleanup(self):
        """Clean up resources."""
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.quit()
            self.batch_thread.wait()
            
        self.batch_thread = None
        self.batch_worker = None
        self.is_processing = False