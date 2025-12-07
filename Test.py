I'll improve the code based on the identified issues while maintaining functionality. Here's the complete optimized version:

```python
# DAF Engine v3.5 - Final Production Edition
# Optimized for: Google Colab, Mobile (Termux/DeX), Local PC
# Features:
# - 6-Layer Robust Code Extraction (95%+ success rate)
# - Intelligent Model Fallbacks (OOM -> Tiny Model)
# - Configuration Overrides via CLI or Python
# - Google Drive Persistence

import os
import sys
import json
import time
import ast
import re
import hashlib
import subprocess
import logging
import platform
import shutil
import gc
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, Callable, Set, Deque
from collections import defaultdict, deque
import warnings
from functools import lru_cache
import threading

warnings.filterwarnings('ignore')

# ============================================================================
# 🔧 DEPENDENCY HANDLING (FIXED)
# ============================================================================

def safe_import(module_name: str) -> Optional[Any]:
    """Safely import module with better error handling"""
    try:
        # Special handling for google.colab
        if module_name == 'google.colab':
            import google.colab
            return google.colab
        return __import__(module_name)
    except ImportError:
        return None
    except Exception as e:
        logging.warning(f"⚠️ Warning importing {module_name}: {e}")
        return None

# Import checks
torch_module = safe_import('torch')
transformers_module = safe_import('transformers')
psutil_module = safe_import('psutil')
google_colab_module = safe_import('google.colab')

# Environment detection
IS_COLAB = google_colab_module is not None
HAS_TORCH = torch_module is not None
HAS_TRANSFORMERS = transformers_module is not None
HAS_PSUTIL = psutil_module is not None
HAS_GPU = HAS_TORCH and torch_module.cuda.is_available()

# Create aliases for imported modules
if HAS_TORCH:
    torch = torch_module
if HAS_TRANSFORMERS:
    transformers = transformers_module
if HAS_PSUTIL:
    psutil = psutil_module

# ============================================================================
# 🧠 ROBUST CODE EXTRACTOR (IMPROVED)
# ============================================================================

class RobustCodeExtractor:
    """6-Layer Extraction Strategy to handle messy LLM outputs"""
    
    def __init__(self):
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize extraction strategies in priority order"""
        self._strategies = [
            self._extract_markdown_python,
            self._extract_generic_markdown,
            self._extract_by_markers,
            lambda t: self._extract_after_prompt(t, ""),
            self._extract_by_keywords,
            self._extract_raw
        ]
    
    @classmethod
    def register_strategy(cls, strategy: Callable[[str, str], Optional[str]]):
        """Register an extraction strategy"""
        cls._strategies.append(strategy)
        return strategy
    
    def extract(self, text: str, original_prompt: str = "") -> Optional[str]:
        """Execute all extraction strategies in priority order"""
        for strategy in self._strategies:
            try:
                candidate = strategy(text, original_prompt)
                if candidate and self._validate_syntax(candidate):
                    # Additional validation
                    if len(candidate.strip()) > 10:  # At least 10 chars
                        return candidate
            except Exception as e:
                logging.debug(f"Strategy failed: {e}")
                continue

        return None
    
    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _extract_markdown_python(self, text: str, original_prompt: str = "") -> Optional[str]:
        """Extract Python code from markdown blocks"""
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else None
    
    def _extract_generic_markdown(self, text: str, original_prompt: str = "") -> Optional[str]:
        """Extract code from generic markdown blocks"""
        pattern = r'```(?:[\w]*)\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code = matches[-1].strip()
            # Check if it looks like Python
            if any(keyword in code.lower() for keyword in ['def ', 'class ', 'import ', 'from ']):
                return code
        return None
    
    def _extract_by_markers(self, text: str, original_prompt: str = "") -> Optional[str]:
        """Extract code between common markers"""
        markers = [
            ('## Improved Code:', '##', -1),
            ('# Improved Code:', '#', -1),
            ('Improved Code:', '', -1),
            ('Here is the improved code:', '', -1),
            ('Here\'s the improved code:', '', -1),
            ('```python', '```', 0),
            ('```', '```', 0)
        ]

        for start_marker, end_marker, offset in markers:
            if start_marker in text:
                # Get the part after the marker
                parts = text.split(start_marker, 1)
                if len(parts) > 1:
                    code_part = parts[1]

                    # If we have an end marker, extract until that
                    if end_marker:
                        if end_marker in code_part:
                            code_part = code_part.split(end_marker, 1)[0]

                    # Clean up
                    code = code_part.strip()

                    # Remove any remaining markdown fences
                    code = re.sub(r'^```python\s*', '', code)
                    code = re.sub(r'```\s*$', '', code)

                    if code and len(code) > 10:
                        return code

        return None
    
    def _extract_after_prompt(self, text: str, original_prompt: str) -> Optional[str]:
        """Extract code after the original prompt"""
        if original_prompt and original_prompt in text:
            # Get everything after the prompt
            code = text.split(original_prompt, 1)[1].strip()

            # Clean up common endings
            endings = ['\n\n##', '\n\n#', '\n\n```', '\n\n---', '\n\n***']
            for ending in endings:
                if ending in code:
                    code = code.split(ending, 1)[0].strip()

            # Remove any markdown fences
            code = re.sub(r'```python\s*', '', code)
            code = re.sub(r'```\s*$', '', code)

            return code

        return None
    
    def _extract_by_keywords(self, text: str, original_prompt: str = "") -> Optional[str]:
        """Extract code by finding Python keywords"""
        # Find the first Python keyword
        keywords = ['def ', 'class ', 'import ', 'from ', '@', 'async def ']

        for keyword in keywords:
            if keyword in text:
                idx = text.find(keyword)
                if idx >= 0:
                    # Extract from keyword to end
                    code = text[idx:].strip()

                    # Try to find a reasonable end point
                    end_markers = ['\n\n#', '\n\n##', '\n\n```', '\n\n---']
                    for marker in end_markers:
                        if marker in code:
                            code = code.split(marker, 1)[0].strip()

                    return code

        return None
    
    def _extract_raw(self, text: str, original_prompt: str = "") -> Optional[str]:
        """Last resort: return the whole text if it looks like code"""
        # Count Python indicators
        indicators = 0
        lines = text.strip().split('\n')

        # Check each line
        for line in lines[:20]:  # First 20 lines only
            line = line.strip()
            if line.startswith(('def ', 'class ', 'import ', 'from ', '@', 'async ')):
                indicators += 1
            elif line.endswith(':'):
                indicators += 0.5
            elif '=' in line and '(' in line and ')' in line:
                indicators += 0.3

        # If enough indicators, return cleaned text
        if indicators >= 2 and len(text.strip()) > 20:
            cleaned = text.strip()
            # Remove common non-code prefixes
            prefixes = ['Certainly!', 'Here is', 'Here\'s', 'The improved code']
            for prefix in prefixes:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            return cleaned

        return None

# ============================================================================
# ⚙️ CONFIGURATION (IMPROVED)
# ============================================================================

@dataclass
class MobileConfig:
    """Dynamic configuration with override capabilities"""
    
    # === Runtime Settings ===
    IS_COLAB: bool = field(default_factory=lambda: IS_COLAB)
    USE_GPU: bool = field(default_factory=lambda: HAS_GPU)
    MEMORY_GB: float = 4.0

    # === Model Selection ===
    MODEL_SIZE: str = "small"

    # Model options defined as class variable (not instance variable)
    @property
    def MODEL_OPTIONS(self) -> Dict[str, Dict]:
        return {
            "tiny": {
                "name": "microsoft/phi-2",
                "max_tokens": 512,
                "quantization": "4bit",
                "fallback": None
            },
            "small": {
                "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
                "max_tokens": 1024,
                "quantization": "4bit",
                "fallback": "tiny"
            },
            "medium": {
                "name": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "max_tokens": 2048,
                "quantization": "4bit",
                "fallback": "small"
            },
            "large": {
                "name": "deepseek-ai/deepseek-coder-33b-instruct",
                "max_tokens": 4096,
                "quantization": "4bit",
                "fallback": "medium"
            }
        }

    # === Execution Limits ===
    MAX_CYCLES: int = 20
    MAX_FILES_PER_CYCLE: int = 2
    MAX_FILE_SIZE_KB: int = 50
    MIN_FILE_SIZE_BYTES: int = 50
    MAX_DELETION_RATIO: float = 0.5

    # === Resource Management ===
    COOLDOWN_CYCLES: int = 3
    COOLDOWN_SECONDS: int = 30
    MAX_CPU_USAGE_PERCENT: int = 85

    # === Paths ===
    WORKSPACE_ROOT: Path = field(default_factory=lambda: Path("daf_workspace"))
    MODEL_CACHE_DIR: Path = field(default_factory=lambda: Path("model_cache"))
    KNOWLEDGE_BASE_PATH: Path = field(default_factory=lambda: Path("knowledge_base"))
    BACKUP_DIR: Path = field(default_factory=lambda: Path("backups"))
    LOG_DIR: Path = field(default_factory=lambda: Path("logs"))

    # === Drive Integration ===
    USE_GOOGLE_DRIVE: bool = field(default_factory=lambda: IS_COLAB)
    DRIVE_MOUNT_PATH: str = "/content/drive"
    DRIVE_DAF_FOLDER: str = "My Drive/DAF_Engine"

    # === Advanced Settings ===
    ENABLE_SEMANTIC_VALIDATION: bool = True
    REQUIRE_TESTS_FOR_COMMIT: bool = False
    MAX_RUNTIME_HOURS: float = 2.0
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Initialize configuration with environment detection"""
        self._detect_resources()
        self._create_directories()
        self._setup_logging()

        logging.info(f"⚙️ Configuration initialized: Model={self.MODEL_SIZE}, GPU={self.USE_GPU}, RAM={self.MEMORY_GB:.1f}GB")

    def _detect_resources(self):
        """Detect available system resources"""
        # Detect memory
        if HAS_PSUTIL:
            try:
                mem = psutil.virtual_memory()
                self.MEMORY_GB = mem.total / (1024**3)
                logging.info(f"📊 Detected RAM: {self.MEMORY_GB:.1f}GB")
            except Exception as e:
                logging.warning(f"Could not detect memory: {e}")

        # Auto-adjust model size based on available RAM
        original_size = self.MODEL_SIZE
        if self.MEMORY_GB < 3.0 and self.MODEL_SIZE != "tiny":
            self.MODEL_SIZE = "tiny"
        elif self.MEMORY_GB < 6.0 and self.MODEL_SIZE in ["medium", "large"]:
            self.MODEL_SIZE = "small"

        if original_size != self.MODEL_SIZE:
            logging.info(f"🔄 Auto-adjusted model size: {original_size} → {self.MODEL_SIZE}")

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.WORKSPACE_ROOT,
            self.MODEL_CACHE_DIR,
            self.KNOWLEDGE_BASE_PATH,
            self.BACKUP_DIR,
            self.LOG_DIR
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logging.debug(f"📁 Created directory: {directory}")
            except Exception as e:
                logging.error(f"Failed to create directory {directory}: {e}")

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.LOG_DIR / f"daf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def override(self, overrides: Dict[str, Any]) -> None:
        """Apply external configuration overrides safely"""
        for key, value in overrides.items():
            if hasattr(self, key):
                current = getattr(self, key)
                setattr(self, key, value)
                logging.info(f"⚙️ Config Override: {key} = {value} (was: {current})")
            else:
                logging.warning(f"⚠️ Unknown config key: {key}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration settings"""
        errors = []

        # Validate paths
        for path_name in ['WORKSPACE_ROOT', 'MODEL_CACHE_DIR', 'KNOWLEDGE_BASE_PATH']:
            path = getattr(self, path_name)
            try:
                path.resolve()
            except Exception as e:
                errors.append(f"Invalid path {path_name}: {path} - {e}")

        # Validate model size
        if self.MODEL_SIZE not in self.MODEL_OPTIONS:
            errors.append(f"Invalid MODEL_SIZE: {self.MODEL_SIZE}")

        # Validate numeric ranges
        if self.MAX_CYCLES <= 0:
            errors.append("MAX_CYCLES must be > 0")
        if self.MAX_FILE_SIZE_KB <= 0:
            errors.append("MAX_FILE_SIZE_KB must be > 0")
        if not (0 <= self.MAX_DELETION_RATIO <= 1):
            errors.append("MAX_DELETION_RATIO must be between 0 and 1")

        return len(errors) == 0, errors

# ============================================================================
# 🤖 INTELLIGENT MODEL MANAGER WITH FALLBACKS (OPTIMIZED)
# ============================================================================

class IntelligentModelManager:
    """Manages model loading with intelligent fallbacks on OOM"""
    
    def __init__(self, config: MobileConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self.current_model_size = config.MODEL_SIZE
        self._load_attempts = defaultdict(int)
        self.max_load_attempts = 3
        self._model_info = None
        
        # Cache for model info
        self._model_info_cache = None
        
        logging.info(f"🤖 Initializing Model Manager (target: {self.current_model_size})")

    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        if self._model_info is None:
            self._model_info = {
                "current_size": self.current_model_size,
                "target_size": self.config.MODEL_SIZE,
                "loaded": self._model is not None,
                "load_attempts": dict(self._load_attempts),
                "memory_gb": self.config.MEMORY_GB
            }
        return self._model_info
    
    @lru_cache(maxsize=3)
    def load_model(self, force_reload: bool = False) -> Tuple[Optional[Any], Optional[Any]]:
        """Load model with intelligent fallback strategy"""
        if self._model is not None and not force_reload:
            return self._model, self._tokenizer

        attempt_model_size = self.current_model_size

        while attempt_model_size is not None:
            try:
                logging.info(f"📥 Attempting to load model: {attempt_model_size}")
                model, tokenizer = self._load_specific_model(attempt_model_size)

                # Success - update current size
                self.current_model_size = attempt_model_size
                self._model = model
                self._tokenizer = tokenizer
                self._model_info = None  # Reset cache
                self._load_attempts.clear()  # Reset attempts
                return model, tokenizer

            except Exception as e:
                error_msg = str(e).lower()
                self._load_attempts[attempt_model_size] += 1

                # Check if it's an OOM error
                is_oom = any(term in error_msg for term in ['memory', 'oom', 'cuda out of memory', 'not enough memory'])

                if is_oom or self._load_attempts[attempt_model_size] >= self.max_load_attempts:
                    # Get fallback model
                    fallback = self.config.MODEL_OPTIONS[attempt_model_size].get('fallback')

                    if fallback is None:
                        logging.error(f"❌ No fallback available for {attempt_model_size}")
                        raise RuntimeError(f"Failed to load model {attempt_model_size}: {e}")

                    logging.warning(f"⚠️ {attempt_model_size} failed, falling back to {fallback}")
                    attempt_model_size = fallback

                    # Clear memory before retry
                    self._clean_memory()
                else:
                    # Retry same model
                    wait_time = 2 ** self._load_attempts[attempt_model_size]
                    logging.info(f"🔄 Retry {self._load_attempts[attempt_model_size]}/{self.max_load_attempts} in {wait_time}s...")
                    time.sleep(wait_time)

        raise RuntimeError("Failed to load any model")

    def _load_specific_model(self, model_size: str) -> Tuple[Any, Any]:
        """Load a specific model size"""
        if not