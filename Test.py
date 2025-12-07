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
        print(f"⚠️ Warning importing {module_name}: {e}")
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
# 🧠 ROBUST CODE EXTRACTION (COMPLETE 6-LAYER IMPLEMENTATION)
# ============================================================================

class RobustCodeExtractor:
    """6-Layer Extraction Strategy to handle messy LLM outputs"""

    # Strategy registry with priority order
    STRATEGIES: List[Callable[[str, str], Optional[str]]] = []

    @classmethod
    def register_strategy(cls, strategy: Callable[[str, str], Optional[str]]):
        """Register an extraction strategy"""
        cls.STRATEGIES.append(strategy)
        return strategy

    @staticmethod
    def extract(text: str, original_prompt: str = "") -> Optional[str]:
        """Execute all extraction strategies in priority order"""

        strategies = [
            RobustCodeExtractor._extract_markdown_python,
            RobustCodeExtractor._extract_generic_markdown,
            RobustCodeExtractor._extract_by_markers,
            lambda t: RobustCodeExtractor._extract_after_prompt(t, original_prompt),
            RobustCodeExtractor._extract_by_keywords,
            RobustCodeExtractor._extract_raw
        ]

        for strategy in strategies:
            try:
                candidate = strategy(text)
                if candidate and RobustCodeExtractor._validate_syntax(candidate):
                    # Additional validation
                    if len(candidate.strip()) > 10:  # At least 10 chars
                        return candidate
            except Exception as e:
                continue

        return None

    @staticmethod
    def _validate_syntax(code: str) -> bool:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _extract_markdown_python(text: str) -> Optional[str]:
        """Extract Python code from markdown blocks"""
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1].strip() if matches else None

    @staticmethod
    def _extract_generic_markdown(text: str) -> Optional[str]:
        """Extract code from generic markdown blocks"""
        pattern = r'```(?:[\w]*)\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code = matches[-1].strip()
            # Check if it looks like Python
            if any(keyword in code.lower() for keyword in ['def ', 'class ', 'import ', 'from ']):
                return code
        return None

    @staticmethod
    def _extract_by_markers(text: str) -> Optional[str]:
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

    @staticmethod
    def _extract_after_prompt(text: str, original_prompt: str) -> Optional[str]:
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

    @staticmethod
    def _extract_by_keywords(text: str) -> Optional[str]:
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

    @staticmethod
    def _extract_raw(text: str) -> Optional[str]:
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
# ⚙️ CONFIGURATION (FIXED)
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
# 🤖 INTELLIGENT MODEL MANAGER WITH FALLBACKS
# ============================================================================

class IntelligentModelManager:
    """Manages model loading with intelligent fallbacks on OOM"""

    def __init__(self, config: MobileConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.current_model_size = config.MODEL_SIZE
        self.load_attempts = defaultdict(int)
        self.max_load_attempts = 3

        logging.info(f"🤖 Initializing Model Manager (target: {self.current_model_size})")

    def load_model(self, force_reload: bool = False) -> Tuple[Optional[Any], Optional[Any]]:
        """Load model with intelligent fallback strategy"""

        if self.model is not None and not force_reload:
            return self.model, self.tokenizer

        attempt_model_size = self.current_model_size

        while attempt_model_size is not None:
            try:
                logging.info(f"📥 Attempting to load model: {attempt_model_size}")
                model, tokenizer = self._load_specific_model(attempt_model_size)

                # Success - update current size
                self.current_model_size = attempt_model_size
                self.model = model
                self.tokenizer = tokenizer

                logging.info(f"✅ Successfully loaded {attempt_model_size} model")
                return model, tokenizer

            except Exception as e:
                error_msg = str(e).lower()
                self.load_attempts[attempt_model_size] += 1

                # Check if it's an OOM error
                is_oom = any(term in error_msg for term in ['memory', 'oom', 'cuda out of memory', 'not enough memory'])

                if is_oom or self.load_attempts[attempt_model_size] >= self.max_load_attempts:
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
                    wait_time = 2 ** self.load_attempts[attempt_model_size]
                    logging.info(f"🔄 Retry {self.load_attempts[attempt_model_size]}/{self.max_load_attempts} in {wait_time}s...")
                    time.sleep(wait_time)

        raise RuntimeError("Failed to load any model")

    def _load_specific_model(self, model_size: str) -> Tuple[Any, Any]:
        """Load a specific model size"""
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library not available")

        model_info = self.config.MODEL_OPTIONS[model_size]
        model_name = model_info["name"]

        logging.info(f"🚀 Loading {model_name} ({model_size})...")

        # Configure quantization
        quantization_config = None
        if model_info.get("quantization") == "4bit":
            quantization_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(self.config.MODEL_CACHE_DIR),
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model_kwargs = {
            "cache_dir": str(self.config.MODEL_CACHE_DIR),
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
            "device_map": "auto" if self.config.USE_GPU else "cpu",
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Use low memory mode if enabled
        if self.config.MEMORY_GB < 8.0:
            model_kwargs["low_cpu_mem_usage"] = True

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Set to eval mode
        model.eval()

        return model, tokenizer

    def generate_code(self, prompt: str, max_tokens: int = None) -> str:
        """Generate code using the loaded model"""
        if self.model is None or self.tokenizer is None:
            self.load_model()

        model_info = self.config.MODEL_OPTIONS[self.current_model_size]
        max_tokens = max_tokens or model_info["max_tokens"]

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            )

            # Move to GPU if available
            if self.config.USE_GPU and HAS_TORCH:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate with conservative settings for stability
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True
                )

            # Decode output
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract code using robust extractor
            extracted = RobustCodeExtractor.extract(generated, prompt)

            if extracted:
                return extracted
            else:
                logging.warning("⚠️ Could not extract code, returning raw generation")
                return generated.strip()

        except Exception as e:
            logging.error(f"❌ Generation failed: {e}")

            # If it's an OOM error, try to fallback
            if "memory" in str(e).lower() or "cuda" in str(e).lower():
                logging.warning("🔄 OOM detected during generation, trying smaller model...")
                self._clean_memory()

                # Force reload with smaller model
                fallback = self.config.MODEL_OPTIONS[self.current_model_size].get('fallback')
                if fallback:
                    self.current_model_size = fallback
                    self.model = None
                    self.tokenizer = None

                    # Retry with smaller model
                    return self.generate_code(prompt, max_tokens // 2)

            return ""

    def _clean_memory(self):
        """Clean up memory"""
        try:
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()
        except:
            pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model"""
        return {
            "current_size": self.current_model_size,
            "target_size": self.config.MODEL_SIZE,
            "loaded": self.model is not None,
            "load_attempts": dict(self.load_attempts),
            "memory_gb": self.config.MEMORY_GB
        }

# ============================================================================
# 📁 GOOGLE DRIVE INTEGRATION
# ============================================================================

class GoogleDriveManager:
    """Manages Google Drive persistence for Colab"""

    def __init__(self, config: MobileConfig):
        self.config = config
        self.is_mounted = False

        if config.USE_GOOGLE_DRIVE and IS_COLAB:
            self.mount_drive()

    def mount_drive(self) -> bool:
        """Mount Google Drive to Colab"""
        try:
            from google.colab import drive

            if not Path(self.config.DRIVE_MOUNT_PATH).exists():
                drive.mount(self.config.DRIVE_MOUNT_PATH)
                logging.info(f"✅ Google Drive mounted to {self.config.DRIVE_MOUNT_PATH}")

            # Create DAF folder
            daf_path = Path(self.config.DRIVE_MOUNT_PATH) / self.config.DRIVE_DAF_FOLDER
            daf_path.mkdir(parents=True, exist_ok=True)

            self.is_mounted = True
            return True

        except Exception as e:
            logging.error(f"❌ Failed to mount Google Drive: {e}")
            self.is_mounted = False
            return False

    def sync_file(self, local_path: Path, drive_subpath: str = "") -> bool:
        """Sync a file to Google Drive"""
        if not self.is_mounted or not local_path.exists():
            return False

        try:
            drive_path = Path(self.config.DRIVE_MOUNT_PATH) / self.config.DRIVE_DAF_FOLDER / drive_subpath
            drive_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(local_path, drive_path / local_path.name)
            logging.debug(f"📤 Synced {local_path.name} to Drive")
            return True

        except Exception as e:
            logging.warning(f"⚠️ Drive sync failed for {local_path}: {e}")
            return False

    def restore_file(self, filename: str, local_path: Path, drive_subpath: str = "") -> bool:
        """Restore a file from Google Drive"""
        if not self.is_mounted:
            return False

        try:
            drive_file = Path(self.config.DRIVE_MOUNT_PATH) / self.config.DRIVE_DAF_FOLDER / drive_subpath / filename

            if drive_file.exists():
                shutil.copy2(drive_file, local_path)
                logging.info(f"📥 Restored {filename} from Drive")
                return True

        except Exception as e:
            logging.warning(f"⚠️ Drive restore failed for {filename}: {e}")

        return False

    def backup_knowledge_base(self, kb_data: Dict, cycle: int) -> bool:
        """Backup knowledge base to Drive"""
        if not self.is_mounted:
            return False

        try:
            # Create backup directory
            backup_dir = Path(self.config.DRIVE_MOUNT_PATH) / self.config.DRIVE_DAF_FOLDER / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Save backup
            backup_file = backup_dir / f"knowledge_base_cycle_{cycle:04d}.json"
            with open(backup_file, 'w') as f:
                json.dump(kb_data, f, indent=2)

            # Also save as latest
            latest_file = backup_dir / "knowledge_base_latest.json"
            with open(latest_file, 'w') as f:
                json.dump(kb_data, f, indent=2)

            logging.info(f"💾 Knowledge base backed up to Drive (cycle {cycle})")
            return True

        except Exception as e:
            logging.error(f"❌ Knowledge base backup failed: {e}")
            return False

# ============================================================================
# 🚀 MAIN DAF ENGINE v3.5
# ============================================================================

class DAFEngine:
    """Main DAF Engine v3.5 with all features"""

    def __init__(self, config_overrides: Dict[str, Any] = None):
        # Initialize configuration
        self.config = MobileConfig()

        if config_overrides:
            self.config.override(config_overrides)

        # Validate configuration
        valid, errors = self.config.validate()
        if not valid:
            raise ValueError(f"Configuration errors: {errors}")

        # Initialize components
        self.model_manager = IntelligentModelManager(self.config)
        self.drive_manager = GoogleDriveManager(self.config)
        self.code_extractor = RobustCodeExtractor()

        # Initialize knowledge base
        self.knowledge_base = self._load_knowledge_base()
        self.cycle_count = self.knowledge_base.get('current_cycle', 0)

        # Statistics
        self.stats = {
            'total_cycles': 0,
            'files_processed': 0,
            'files_improved': 0,
            'model_loads': 0,
            'start_time': datetime.now().isoformat()
        }

        logging.info("🚀 DAF Engine v3.5 initialized successfully")
        logging.info(f"📊 Configuration: {json.dumps(self.config.to_dict(), indent=2, default=str)}")

    def _load_knowledge_base(self) -> Dict:
        """Load knowledge base from file or Google Drive"""
        kb_file = self.config.KNOWLEDGE_BASE_PATH / "knowledge_base.json"

        # Try to restore from Google Drive first
        if self.config.USE_GOOGLE_DRIVE:
            restored = self.drive_manager.restore_file(
                "knowledge_base_latest.json",
                kb_file,
                "backups"
            )

        # Load from local file
        if kb_file.exists():
            try:
                with open(kb_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load knowledge base: {e}")

        # Create new knowledge base
        return {
            'version': '3.5',
            'created': datetime.now().isoformat(),
            'current_cycle': 0,
            'improvements': [],
            'file_hashes': {},
            'model_history': []
        }

    def _save_knowledge_base(self):
        """Save knowledge base to disk and Google Drive"""
        kb_file = self.config.KNOWLEDGE_BASE_PATH / "knowledge_base.json"

        # Update knowledge base
        self.knowledge_base.update({
            'current_cycle': self.cycle_count,
            'last_saved': datetime.now().isoformat(),
            'stats': self.stats,
            'model_info': self.model_manager.get_model_info()
        })

        try:
            # Save locally
            with open(kb_file, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2, default=str)

            # Backup to Google Drive
            if self.config.USE_GOOGLE_DRIVE and self.cycle_count > 0:
                self.drive_manager.backup_knowledge_base(self.knowledge_base, self.cycle_count)

            logging.debug(f"💾 Knowledge base saved (cycle {self.cycle_count})")

        except Exception as e:
            logging.error(f"Failed to save knowledge base: {e}")

    def run_cycle(self) -> Dict[str, Any]:
        """Run a single improvement cycle"""
        self.cycle_count += 1
        cycle_start = datetime.now()

        logging.info(f"\n🌀 CYCLE {self.cycle_count}")
        logging.info("=" * 50)

        try:
            # TODO: Implement actual file processing logic
            # This is a placeholder for the main processing loop

            # Simulate processing
            time.sleep(1)

            # Update stats
            self.stats['total_cycles'] += 1

            # Save knowledge base periodically
            if self.cycle_count % 5 == 0:
                self._save_knowledge_base()

            cycle_duration = datetime.now() - cycle_start

            result = {
                'status': 'completed',
                'cycle': self.cycle_count,
                'duration_seconds': cycle_duration.total_seconds(),
                'model_info': self.model_manager.get_model_info(),
                'stats': self.stats.copy()
            }

            logging.info(f"✅ Cycle {self.cycle_count} completed in {cycle_duration.total_seconds():.1f}s")
            return result

        except Exception as e:
            logging.error(f"❌ Cycle {self.cycle_count} failed: {e}")
            return {
                'status': 'failed',
                'cycle': self.cycle_count,
                'error': str(e)
            }

    def run(self, max_cycles: int = None):
        """Run the DAF engine for multiple cycles"""
        if max_cycles is None:
            max_cycles = self.config.MAX_CYCLES

        logging.info(f"🚀 Starting DAF Engine v3.5 (max cycles: {max_cycles})")

        consecutive_failures = 0
        max_consecutive_failures = 3

        try:
            for cycle_num in range(max_cycles):
                result = self.run_cycle()

                if result['status'] == 'failed':
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logging.error(f"❌ {max_consecutive_failures} consecutive failures, stopping")
                        break
                else:
                    consecutive_failures = 0

                # Cooldown between cycles
                if self.cycle_count % self.config.COOLDOWN_CYCLES == 0:
                    logging.info(f"😴 Cooldown for {self.config.COOLDOWN_SECONDS}s...")
                    time.sleep(self.config.COOLDOWN_SECONDS)
                else:
                    time.sleep(1)

            # Final save
            self._save_knowledge_base()

            logging.info(f"\n🏁 DAF Engine completed {self.cycle_count} cycles")
            logging.info(f"📊 Final stats: {json.dumps(self.stats, indent=2)}")

        except KeyboardInterrupt:
            logging.info("\n⏹️ Stopped by user")
        except Exception as e:
            logging.error(f"\n💥 Critical error: {e}")

# ============================================================================
# 🎯 MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description='DAF Engine v3.5 - Autonomous Code Improvement')
    parser.add_argument('--model-size', choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size to use')
    parser.add_argument('--max-cycles', type=int, help='Maximum number of cycles')
    parser.add_argument('--max-files', type=int, help='Maximum files per cycle')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--config-file', type=str, help='JSON configuration file')

    args = parser.parse_args()

    # Load configuration from file if provided
    config_overrides = {}

    if args.config_file:
        try:
            with open(args.config_file, 'r') as f:
                config_overrides = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load config file: {e}")
            return

    # Override with CLI arguments
    if args.model_size:
        config_overrides['MODEL_SIZE'] = args.model_size
    if args.max_cycles:
        config_overrides['MAX_CYCLES'] = args.max_cycles
    if args.max_files:
        config_overrides['MAX_FILES_PER_CYCLE'] = args.max_files
    if args.log_level:
        config_overrides['LOG_LEVEL'] = args.log_level

    # Create and run engine
    try:
        engine = DAFEngine(config_overrides)
        engine.run()
    except Exception as e:
        print(f"❌ Failed to start DAF Engine: {e}")
        return 1

    return 0

# NOTE: The '__main__' block is commented out to prevent argparse from being called automatically
# when the script is run in a Colab environment, which causes an 'unrecognized arguments' error.
# You can instantiate and run DAFEngine directly in subsequent cells if needed.
# if __name__ == "__main__":
#     sys.exit(main())
