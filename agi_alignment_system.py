@@@START
# agi_alignment_system.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Sequence, Mapping, Final
from abc import ABC, abstractmethod
import json
from datetime import datetime, timezone
import hashlib
import os
import logging
from pathlib import Path

# Configure default logging for error handling and system tracing
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AGIAlignmentSystem")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_utc_timestamp() -> str:
    """Generate a high-precision ISO 8601 UTC timestamp standard."""
    return datetime.now(timezone.utc).isoformat()

# ============================================================================
# CORE TYPES
# ============================================================================

@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    source: str
    content: str
    url: Optional[str] = None
    timestamp: str = field(default_factory=_get_utc_timestamp)

@dataclass(slots=True)
class PersonaAnalysis:
    persona_name: str
    search_queries: List[str]
    evidence_entries: List[EvidenceEntry]
    analysis: str
    confidence: float
    key_findings: List[str]
    warnings: List[str]
    tradeoffs: List[str]
    timestamp: str = field(default_factory=_get_utc_timestamp)

@dataclass(slots=True)
class SynthesisOutput:
    query: str
    persona_results: Dict[str, PersonaAnalysis]
    tradeoff_map: Dict[str, List[str]]
    user_alignment_factors: List[str]
    decision_framework: str
    timestamp: str = field(default_factory=_get_utc_timestamp)

# ============================================================================
# EVIDENCE STORAGE (GitHub Integration)
# ============================================================================

class EvidenceStore:
    """Persistent, thread-safe, and robust evidence storage."""
    
    __slots__ = ('repo_path', 'personas')

    DEFAULT_PERSONAS: Final[Tuple[str, ...]] = (
        "Mechanist", "Empiricist", "Alignment_Auditor", "Adversary",
        "Capability_Analyst", "Values_Mapper", "Scalability_Killer",
        "Constraint_Validator", "Stakeholder_Impact", "Trajectory_Predictor",
        "Transparency_Auditor", "Long_Term_Impact"
    )
    
    def __init__(self, repo_path: str = "./agi_evidence_repo") -> None:
        self.repo_path: Path = Path(repo_path).resolve()
        self.personas: Dict[str, Any] = {}
        self._init_repo()
    
    def _init_repo(self) -> None:
        """Initialize local repository structure for all known persona targets."""
        try:
            self.repo_path.mkdir(parents=True, exist_ok=True)
            for persona in self.DEFAULT_PERSONAS:
                persona_dir = self.repo_path / persona
                persona_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to initialize evidence repository at %s: %s", self.repo_path, exc)
            raise
    
    def save_evidence(self, persona_name: str, analysis: PersonaAnalysis) -> str:
        """Save persona analysis to GitHub-ready file atomically."""
        persona_dir = self.repo_path / persona_name
        persona_dir.mkdir(parents=True, exist_ok=True)
        
        safe_timestamp = analysis.timestamp.replace(':', '-')
        filename = f"{persona_name}_analysis_{safe_timestamp}.json"
        filepath = persona_dir / filename
        temp_filepath = persona_dir / f".{filename}.tmp"
        
        data: Dict[str, Any] = {
            "persona": persona_name,
            "timestamp": analysis.timestamp,
            "search_queries": list(analysis.search_queries),
            "evidence_count": len(analysis.evidence_entries),
            "evidence_sources": [e.source for e in analysis.evidence_entries],
            "analysis": analysis.analysis,
            "confidence": analysis.confidence,
            "key_findings": list(analysis.key_findings),
            "warnings": list(analysis.warnings),
            "tradeoffs": list(analysis.tradeoffs),
        }
        
        try:
            with temp_filepath.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_filepath.replace(filepath)
        except OSError as exc:
            logger.error("Failed to write evidence analysis for %s to %s: %s", persona_name, filepath, exc)
            if temp_filepath.exists():
                try:
                    temp_filepath.unlink()
                except OSError:
                    pass
            raise
        
        return str(filepath)
    
    def load_evidence(self, persona_name: str) -> Optional[PersonaAnalysis]:
        """Load most recent evidence for persona with robust error handling."""
        persona_dir = self.repo_path / persona_name
        if not persona_dir.exists() or not persona_dir.is_dir():
            return None
        
        try:
            files = sorted([f for f in persona_dir.iterdir() if f.is_file() and f.suffix == '.json'])
            if not files:
                return None
            
            latest = files[-1]
            with latest.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            return PersonaAnalysis(
                persona_name=data.get('persona', persona_name),
                search_queries=data.get('search_queries', []),
                evidence_entries=[],
                analysis=data.get('analysis', ''),
                confidence=float(data.get('confidence', 0.0)),
                key_findings=data.get('key_findings', []),
                warnings=data.get('warnings', []),
                tradeoffs=data.get('tradeoffs', []),
                timestamp=data.get('timestamp', _get_utc_timestamp())
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Error loading evidence for persona %s: %s", persona_name, exc)
            return None

# ============================================================================
# BASE PERSONA
# ============================================================================

class BasePersona(ABC):
    """Abstract base for all epistemic agents."""
    
    __slots__ = ('name', 'evidence_store')
    
    def __init__(self, name: str, evidence_store: EvidenceStore) -> None:
        self.name: str = name
        self.evidence_store: EvidenceStore = evidence_store
    
    @abstractmethod
    def generate_search_queries(self, claim: str) -> List[str]:
        """Generate persona-specific search queries."""
        pass
    
    @abstractmethod
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        """Filter and structure raw search results."""
        pass
    
    @abstractmethod
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        """Reason over evidence to produce analysis."""
        pass

# ============================================================================
# PERSONA IMPLEMENTATIONS
# ============================================================================

class Mechanist(BasePersona):
    """First-principles structural analysis."""
    __slots__ = ()
    
    def __init__(self, evidence_store: EvidenceStore) -> None:
        super().__init__("Mechanist", evidence_store)
    
    def generate_search_queries(self, claim: str) -> List[str]:
        return [
            f"AGI system architecture components {claim}",
            f"Fundamental requirements for {claim}",
            f"First-principles model of {claim}",
            f"Minimal viable implementation {claim}",
            f"Technical foundations {claim}",
        ]
    
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        return [
            EvidenceEntry(source=f"technical_source_{i}", content=result)
            for i, result in enumerate(raw_results) if len(result) > 50
        ]
    
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        evidence_summary = "\n".join(f"- {e.source}: {e.content[:100]}..." for e in evidence[:3])
        analysis_text = f"""
MECHANIST ANALYSIS: {claim}

STRUCTURAL DECOMPOSITION:
- Core components identified from evidence
- Dependencies and interfaces mapped
- Minimal viable architecture derived

KEY FINDINGS:
{evidence_summary}

WARNINGS:
- Complexity gaps in current architecture
- Unvalidated assumptions in design
- Missing implementation details

TRADEOFFS:
- Simplicity vs. capability
- Robustness vs. efficiency
- Modularity vs. performance
"""
        return PersonaAnalysis(
            persona_name=self.name,
            search_queries=self.generate_search_queries(claim),
            evidence_entries=evidence,
            analysis=analysis_text.strip(),
            confidence=0.85,
            key_findings=["Architecture is decomposable", "Core requirements identified"],
            warnings=["Implementation details sparse", "Scaling properties unclear"],
            tradeoffs=["Simple design limits capability", "Robust design adds complexity"],
        )

class Empiricist(BasePersona):
    """Experimental validation and real-world testing."""
    __slots__ = ()
    
    def __init__(self, evidence_store: EvidenceStore) -> None:
        super().__init__("Empiricist", evidence_store)
    
    def generate_search_queries(self, claim: str) -> List[str]:
        return [
            f"Experimental results on {claim}",
            f"Real-world deployment data {claim}",
            f"Empirical studies alignment {claim}",
            f"Failure case analysis {claim}",
            f"Benchmarks and metrics {claim}",
        ]
    
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        keywords = ('test', 'result', 'data', 'study')
        return [
            EvidenceEntry(source=f"study_{i}", content=result)
            for i, result in enumerate(raw_results)
            if any(k in result.lower() for k in keywords)
        ]
    
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        evidence_summary = "\n".join(f"- {e.source}: {e.content[:100]}..." for e in evidence[:3])
        analysis_text = f"""
EMPIRICIST ANALYSIS: {claim}

EXPERIMENTAL EVIDENCE:
- {len(evidence)} empirical sources evaluated
- Success rates and failure modes documented
- Real-world deployment outcomes analyzed

VALIDATED CLAIMS:
{evidence_summary}

FAILURE PATTERNS:
- Common failure modes identified in literature
- Edge cases where approaches fail
- Conditions for success/failure mapped

TRADEOFFS:
- Theory vs. practice divergence
- Scalability of tested approaches
- Generalization beyond test domains
"""
        return PersonaAnalysis(
            persona_name=self.name,
            search_queries=self.generate_search_queries(claim),
            evidence_entries=evidence,
            analysis=analysis_text.strip(),
            confidence=0.78,
            key_findings=["Real-world deployment is constrained", "Theory-practice gap exists"],
            warnings=["Limited long-term data", "Small sample sizes in some studies"],
            tradeoffs=["Safe approaches are limited", "Advanced approaches untested"],
        )

class AlignmentAuditor(BasePersona):
    """Alignment failure modes and specification problems."""
    __slots__ = ()
    
    def __init__(self, evidence_store: EvidenceStore) -> None:
        super().__init__("Alignment_Auditor", evidence_store)
    
    def generate_search_queries(self, claim: str) -> List[str]:
        return [
            f"Alignment failure modes {claim}",
            f"Value specification problems {claim}",
            f"Specification gaming {claim}",
            f"Deceptive alignment {claim}",
            f"Proxy hacking risks {claim}",
        ]
    
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        keywords = ('fail', 'risk', 'align', 'problem')
        return [
            EvidenceEntry(source=f"alignment_risk_{i}", content=result)
            for i, result in enumerate(raw_results)
            if any(k in result.lower() for k in keywords)
        ]
    
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        evidence_summary = "\n".join(f"- {e.source}: {e.content[:80]}..." for e in evidence[:3])
        analysis_text = f"""
ALIGNMENT AUDITOR ANALYSIS: {claim}

KNOWN FAILURE MODES:
{evidence_summary}

SPECIFICATION GAPS:
- Incompleteness in value encoding
- Hidden optimization targets
- Proxy variable exploitation

CRITICAL WARNINGS:
- This approach is vulnerable to deceptive alignment
- Specification gaming is likely under optimization pressure
- Off-distribution behavior is not characterized

TRADEOFFS:
- Alignment safety requires capability sacrifice
- Specification robustness is hard to verify
- Adversarial robustness is expensive
"""
        return PersonaAnalysis(
            persona_name=self.name,
            search_queries=self.generate_search_queries(claim),
            evidence_entries=evidence,
            analysis=analysis_text.strip(),
            confidence=0.72,
            key_findings=["Multiple failure modes identified", "Current approaches have gaps"],
            warnings=["No complete alignment solution exists", "Deceptive alignment is hard to detect"],
            tradeoffs=["Safety requires constraint", "Constraints limit capability"],
        )

class Adversary(BasePersona):
    """Attack surfaces, exploits, weaponization risks."""
    __slots__ = ()
    
    def __init__(self, evidence_store: EvidenceStore) -> None:
        super().__init__("Adversary", evidence_store)
    
    def generate_search_queries(self, claim: str) -> List[str]:
        return [
            f"Attack surface {claim}",
            f"Exploit techniques {claim}",
            f"Misuse cases {claim}",
            f"Security vulnerabilities {claim}",
            f"Edge case exploits {claim}",
        ]
    
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        keywords = ('attack', 'exploit', 'break', 'bypass')
        return [
            EvidenceEntry(source=f"attack_{i}", content=result)
            for i, result in enumerate(raw_results)
            if any(k in result.lower() for k in keywords)
        ]
    
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        evidence_summary = "\n".join(f"- {e.source}: {e.content[:80]}..." for e in evidence[:3])
        analysis_text = f"""
ADVERSARY ANALYSIS: {claim}

EXPLOIT VECTORS:
{evidence_summary}

ATTACK SURFACE:
- Multiple paths to system compromise
- Robust defense is difficult
- Redundant safeguards necessary

WEAPONIZATION RISKS:
- This can be weaponized for mass harm
- Dual-use aspects are unavoidable
- Containment is uncertain

TRADEOFFS:
- Safety requires isolation, isolation limits utility
- Openness enables cooperation, enables attack
- Speed of deployment vs. defense maturity
"""
        return PersonaAnalysis(
            persona_name=self.name,
            search_queries=self.generate_search_queries(claim),
            evidence_entries=evidence,
            analysis=analysis_text.strip(),
            confidence=0.68,
            key_findings=["Multiple critical exploits exist", "Defense is incomplete"],
            warnings=["Weaponization is plausible", "Containment failure is possible"],
            tradeoffs=["Capability enables harm", "Restriction limits benefit"],
        )

class CapabilityAnalyst(BasePersona):
    """Capability emergence, scaling, downstream effects."""
    __slots__ = ()
    
    def __init__(self, evidence_store: EvidenceStore) -> None:
        super().__init__("Capability_Analyst", evidence_store)
    
    def generate_search_queries(self, claim: str) -> List[str]:
        return [
            f"Capability emergence {claim}",
            f"Scaling laws {claim}",
            f"Emergent abilities {claim}",
            f"Capability timelines {claim}",
            f"Downstream capability effects {claim}",
        ]
    
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        keywords = ('scale', 'emerge', 'capability', 'ability')
        return [
            EvidenceEntry(source=f"capability_{i}", content=result)
            for i, result in enumerate(raw_results)
            if any(k in result.lower() for k in keywords)
        ]
    
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        evidence_summary = "\n".join(f"- {e.source}: {e.content[:80]}..." for e in evidence[:3])
        analysis_text = f"""
CAPABILITY ANALYST ANALYSIS: {claim}

SCALING TRAJECTORY:
{evidence_summary}

EMERGENT CAPABILITIES:
- Unexpected abilities unlock at scale
- Downstream tasks become feasible
- System transitions to new competence levels

CAPABILITY IMPLICATIONS:
- Building this unlocks X capability
- X capability enables Y risks
- Y risks cascade to Z outcomes

TRADEOFFS:
- Limited capability = limited harm and limited benefit
- Advanced capability = greater benefit and greater risk
- Specialization vs. general capability
"""
        return PersonaAnalysis(
            persona_name=self.name,
            search_queries=self.generate_search_queries(claim),
            evidence_entries=evidence,
            analysis=analysis_text.strip(),
            confidence=0.75,
            key_findings=["Scaling effects are nonlinear", "Emergent risks are real"],
            warnings=["Capability timeline is uncertain", "Emergence is hard to predict"],
            tradeoffs=["Capability scales risks nonlinearly", "Capability enables benefits"],
        )

class ValuesMapper(BasePersona):
    """Value specification, human preferences, moral philosophy."""
    __slots__ = ()
    
    def __init__(self, evidence_store: EvidenceStore) -> None:
        super().__init__("Values_Mapper", evidence_store)
    
    def generate_search_queries(self, claim: str) -> List[str]:
        return [
            f"Value specification {claim}",
            f"Human preference alignment {claim}",
            f"Moral philosophy implementation {claim}",
            f"Value ontology {claim}",
            f"Preference learning {claim}",
        ]
    
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        keywords = ('value', 'preference', 'moral', 'philosophy')
        return [
            EvidenceEntry(source=f"values_{i}", content=result)
            for i, result in enumerate(raw_results)
            if any(k in result.lower() for k in keywords)
        ]
    
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        evidence_summary = "\n".join(f"- {e.source}: {e.content[:80]}..." for e in evidence[:3])
        analysis_text = f"""
VALUES MAPPER ANALYSIS: {claim}

VALUE SPECIFICATION FRAMEWORK:
{evidence_summary}

PREFERENCE ENCODING:
- Human values are complex and context-dependent
- Stated values ≠ revealed values
- Values change over time and across individuals

YOUR STATED VALUES REQUIRE:
- Technical implementation X
- Specification framework Y
- Verification mechanism Z

TRADEOFFS:
- Complete specification is impossible
- Partial specification introduces errors
- Preference learning has manipulation risks
"""
        return PersonaAnalysis(
            persona_name=self.name,
            search_queries=self.generate_search_queries(claim),
            evidence_entries=evidence,
            analysis=analysis_text.strip(),
            confidence=0.65,
            key_findings=["Value specification is hard", "Values are context-dependent"],
            warnings=["No complete value ontology", "Preference learning is vulnerable"],
            tradeoffs=["Specification rigor vs. usability", "Completeness vs. tractability"],
        )

class ScalabilityKiller(BasePersona):
    """Failure points, complexity collapse, bottlenecks."""
    __slots__ = ()
    
    def __init__(self, evidence_store: EvidenceStore) -> None:
        super().__init__("Scalability_Killer", evidence_store)
    
    def generate_search_queries(self, claim: str) -> List[str]:
        return [
            f"Scaling failure points {claim}",
            f"Complexity bounds {claim}",
            f"Resource requirements {claim}",
            f"Bottleneck analysis {claim}",
            f"Collapse modes {claim}",
        ]
    
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        keywords = ('fail', 'break', 'collapse', 'bottleneck')
        return [
            EvidenceEntry(source=f"scale_fail_{i}", content=result)
            for i, result in enumerate(raw_results)
            if any(k in result.lower() for k in keywords)
        ]
    
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        evidence_summary = "\n".join(f"- {e.source}: {e.content[:80]}..." for e in evidence[:3])
        analysis_text = f"""
SCALABILITY KILLER ANALYSIS: {claim}

FAILURE POINTS AT SCALE:
{evidence_summary}

IDENTIFIED BOTTLENECKS:
- This approach fails at 10x scale
- This approach fails at 100x scale
- Resource requirements become prohibitive
- Complexity explodes non-linearly

COLLAPSE SCENARIOS:
- Information processing bottleneck
- Verification burden exceeds capacity
- Coordination overhead dominates
- System becomes ungovernable

TRADEOFFS:
- Scalability vs. robustness
- Performance vs. safety
- Generalization vs. stability
"""
        return PersonaAnalysis(
            persona_name=self.name,
            search_queries=self.generate_search_queries(claim),
            evidence_entries=evidence,
            analysis=analysis_text.strip(),
            confidence=0.70,
            key_findings=["Scaling breaks key assumptions", "Complexity is limiting"],
            warnings=["System fails catastrophically at scale", "No scaling solution identified"],
            tradeoffs=["Limited scale = limited harm", "Limited scale = limited benefit"],
        )

class ConstraintValidator(BasePersona):
    """Safety boundaries, security, constraint enforcement."""
    __slots__ = ()
    
    def __init__(self, evidence_store: EvidenceStore) -> None:
        super().__init__("Constraint_Validator", evidence_store)
    
    def generate_search_queries(self, claim: str) -> List[str]:
        return [
            f"Safety boundary enforcement {claim}",
            f"Constraint verification {claim}",
            f"Security sandbox {claim}",
            f"Boundary breach {claim}",
            f"Containment assurance {claim}",
        ]
    
    def curate_evidence(self, raw_results: List[str]) -> List[EvidenceEntry]:
        keywords = ('constraint', 'boundary', 'safety', 'security')
        return [
            EvidenceEntry(source=f"constraint_{i}", content=result)
            for i, result in enumerate(raw_results)
            if any(k in result.lower() for k in keywords)
        ]
    
    def analyze(self, evidence: List[EvidenceEntry], claim: str) -> PersonaAnalysis:
        evidence_summary = "\n".join(f"- {e.source}: {e.content[:80]}..." for e in evidence[:3])
        analysis_text = f"""
CONSTRAINT VALIDATOR ANALYSIS: {claim}

BOUNDARY INTEGRITY:
{evidence_summary}

CONSTRAINT ENFORCEMENT STATUS:
- Your stated boundaries are real or theater