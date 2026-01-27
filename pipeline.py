"""
HNSCC Pipeline Orchestrator

Main entry point for running the complete image processing pipeline.
Supports checkpoint-based resume and step control.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import Config
from .core.base import CheckpointManager, setup_logger
from .core.utils import scan_input_directory
from .modules import (
    IlluminationProcessor,
    AlignmentProcessor,
    QuenchSubtractor,
    StitchingProcessor
)


class Pipeline:
    """
    Main pipeline orchestrator for HNSCC image processing.
    
    Steps:
    1. illuminate - Illumination correction
    2. align - FOV-level alignment
    3. subtract - Quench subtraction
    4. stitch - Tile stitching with ASHLAR
    """
    
    STEPS = ['illuminate', 'align', 'subtract', 'stitch']
    
    WORKFLOWS = {
        'full': ['illuminate', 'align', 'subtract', 'stitch'],
        'no_align': ['illuminate', 'subtract', 'stitch'],
        'stitch_only': ['stitch'],
        'preprocess': ['illuminate', 'align', 'subtract'],
    }
    
    def __init__(self, config: Optional[Config] = None, config_file: Optional[Path] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration object
            config_file: Path to configuration file (alternative to config)
        """
        if config_file:
            self.config = Config(config_file)
        elif config:
            self.config = config
        else:
            self.config = Config()
        
        self.logger = setup_logger("Pipeline")
        
        # Initialize processors (lazy)
        self._illumination = None
        self._alignment = None
        self._subtraction = None
        self._stitching = None
    
    @property
    def illumination(self) -> IlluminationProcessor:
        if self._illumination is None:
            mode = self.config.get('illumination.mode', 'simple')
            self._illumination = IlluminationProcessor(self.config, mode=mode)
        return self._illumination
    
    @property
    def alignment(self) -> AlignmentProcessor:
        if self._alignment is None:
            self._alignment = AlignmentProcessor(self.config)
        return self._alignment
    
    @property
    def subtraction(self) -> QuenchSubtractor:
        if self._subtraction is None:
            mode = self.config.get('subtraction.mode', 'simple')
            self._subtraction = QuenchSubtractor(self.config, mode=mode)
        return self._subtraction
    
    @property
    def stitching(self) -> StitchingProcessor:
        if self._stitching is None:
            self._stitching = StitchingProcessor(self.config)
        return self._stitching
    
    def run(self, 
            input_dir: Path, 
            output_dir: Path,
            sample_name: str = None,
            marker_names: List[str] = None,
            workflow: str = None,
            start_step: str = None,
            end_step: str = None,
            resume: bool = False) -> Dict[str, Any]:
        """
        Run the pipeline.
        
        Args:
            input_dir: Input directory containing cycle/quench subdirs
            output_dir: Output directory
            sample_name: Sample name (auto-detected if not provided)
            marker_names: List of channel names for OME-TIFF
            workflow: Predefined workflow name ('full', 'no_align', etc.)
            start_step: Step to start from
            end_step: Step to end at
            resume: Whether to resume from checkpoint
            
        Returns:
            Dict with pipeline results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup checkpoint manager
        checkpoint_dir = output_dir / 'checkpoints'
        checkpoint = CheckpointManager(checkpoint_dir)
        
        # Determine steps to execute
        steps = self._resolve_steps(workflow, start_step, end_step, 
                                    resume, checkpoint)
        
        self.logger.info(f"Pipeline started")
        self.logger.info(f"  Input: {input_dir}")
        self.logger.info(f"  Output: {output_dir}")
        self.logger.info(f"  Steps: {steps}")
        
        # Auto-detect sample name
        if not sample_name:
            scan_result = scan_input_directory(input_dir)
            sample_name = scan_result.get('sample_name', 'sample')
        
        self.logger.info(f"  Sample: {sample_name}")
        
        # Execute steps
        results = {}
        current_input = input_dir
        
        for step in steps:
            step_output = output_dir / step
            step_output.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Step: {step}")
            self.logger.info(f"{'='*60}")
            
            checkpoint.save_state(step, 'in_progress')
            
            try:
                if step == 'illuminate':
                    result = self.illumination.process(current_input, step_output)
                    current_input = step_output
                    
                elif step == 'align':
                    result = self.alignment.process(current_input, step_output)
                    current_input = step_output
                    
                elif step == 'subtract':
                    result = self.subtraction.process(current_input, step_output)
                    current_input = step_output
                    
                elif step == 'stitch':
                    result = self.stitching.process(
                        current_input, step_output,
                        sample_name=sample_name,
                        marker_names=marker_names
                    )
                
                results[step] = result
                checkpoint.save_state(step, 'completed', result.get('stats'))
                
            except Exception as e:
                self.logger.error(f"Step {step} failed: {e}")
                checkpoint.save_state(step, 'failed', {'error': str(e)})
                results[step] = {'status': 'failed', 'error': str(e)}
                break
        
        # Summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Pipeline completed")
        for step, result in results.items():
            status = result.get('status', 'unknown')
            self.logger.info(f"  {step}: {status}")
        self.logger.info(f"{'='*60}")
        
        return {
            'status': 'success' if all(r.get('status') == 'success' for r in results.values()) else 'partial',
            'steps': results,
            'output_dir': str(output_dir)
        }
    
    def _resolve_steps(self, workflow: str, start_step: str, end_step: str,
                       resume: bool, checkpoint: CheckpointManager) -> List[str]:
        """Resolve which steps to execute"""
        # Use workflow if specified
        if workflow:
            if workflow not in self.WORKFLOWS:
                raise ValueError(f"Unknown workflow: {workflow}. "
                               f"Available: {list(self.WORKFLOWS.keys())}")
            steps = self.WORKFLOWS[workflow].copy()
        else:
            steps = self.STEPS.copy()
        
        # Handle resume
        if resume:
            last_completed = checkpoint.get_resume_step()
            if last_completed:
                try:
                    idx = steps.index(last_completed)
                    steps = steps[idx + 1:]  # Start from next step
                    self.logger.info(f"Resuming from after '{last_completed}'")
                except ValueError:
                    pass
        
        # Handle start_step
        if start_step:
            if start_step not in self.STEPS:
                raise ValueError(f"Invalid start step: {start_step}")
            start_idx = self.STEPS.index(start_step)
            steps = [s for s in steps if self.STEPS.index(s) >= start_idx]
        
        # Handle end_step
        if end_step:
            if end_step not in self.STEPS:
                raise ValueError(f"Invalid end step: {end_step}")
            end_idx = self.STEPS.index(end_step)
            steps = [s for s in steps if self.STEPS.index(s) <= end_idx]
        
        return steps
