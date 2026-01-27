"""
Base classes and utilities for HNSCC pipeline processors
"""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from ..config import Config


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with standard formatting"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class BaseProcessor(ABC):
    """
    Abstract base class for all pipeline processors.
    
    Provides common functionality for:
    - Configuration access
    - Logging
    - Input/output directory management
    """
    
    def __init__(self, config: Config):
        """
        Initialize processor with configuration.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, input_dir: Path, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Process data from input_dir to output_dir.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            **kwargs: Additional processor-specific arguments
            
        Returns:
            Dict with processing results and status
        """
        pass
    
    def _ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists, create if needed"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _log_start(self, step_name: str, input_dir: Path):
        """Log processing start"""
        self.logger.info(f"Starting {step_name}")
        self.logger.info(f"  Input: {input_dir}")
    
    def _log_complete(self, step_name: str, output_dir: Path, stats: Dict = None):
        """Log processing completion"""
        self.logger.info(f"Completed {step_name}")
        self.logger.info(f"  Output: {output_dir}")
        if stats:
            for key, value in stats.items():
                self.logger.info(f"  {key}: {value}")


class CheckpointManager:
    """
    Manage pipeline checkpoints for resume support.
    
    Saves state after each step to enable resuming from interruptions.
    """
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "pipeline_state.json"
        self.logger = setup_logger("CheckpointManager")
    
    def save_state(self, step: str, status: str, metadata: Optional[Dict] = None):
        """
        Save current pipeline state.
        
        Args:
            step: Current step name
            status: Step status ('completed', 'failed', 'in_progress')
            metadata: Optional additional metadata
        """
        state = self.load_state() or {'completed_steps': [], 'history': []}
        
        step_record = {
            'step': step,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        if status == 'completed':
            if step not in state['completed_steps']:
                state['completed_steps'].append(step)
            state['current_step'] = None
        else:
            state['current_step'] = step
        
        state['history'].append(step_record)
        state['last_updated'] = datetime.now().isoformat()
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.debug(f"Saved checkpoint: {step} - {status}")
    
    def load_state(self) -> Optional[Dict]:
        """
        Load saved pipeline state.
        
        Returns:
            State dictionary or None if no checkpoint exists
        """
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_resume_step(self) -> Optional[str]:
        """
        Get the step to resume from.
        
        Returns:
            Name of last completed step, or None if no checkpoint
        """
        state = self.load_state()
        if state and state.get('completed_steps'):
            return state['completed_steps'][-1]
        return None
    
    def is_step_completed(self, step: str) -> bool:
        """Check if a step has been completed"""
        state = self.load_state()
        if state:
            return step in state.get('completed_steps', [])
        return False
    
    def clear(self):
        """Clear all checkpoints"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            self.logger.info("Cleared checkpoint file")
