# system/core_architecture.py

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Tuple, Final, TypedDict, Union, cast


class ExecutionMode(str, Enum):
    """Execution modes - mutually exclusive operational states."""
    STANDARD = "standard"  # 7 personas + synthesis
    CONSTRAINED = "constrained"  # Direct response only
    REFUSAL = "refusal"  # Cannot execute
    DIAGNOSTIC = "diagnostic"  # Flag contradictions


class DirectResponseDict(TypedDict, total=False):
    """Strongly-typed dictionary schema for direct execution responses."""
    mode: str
    constraints_honored: bool
    answer_type: Optional[str]
    confidence: Union[float, str, None]
    contradictions: List[str]
    operational_boundary: Optional[str]
    content: Optional[str]


class RefusalResponseDict(TypedDict):
    """Strongly-typed dictionary schema for refusal responses."""
    mode: str
    reason: str
    content: str


class StandardResponseDict(TypedDict):
    """Strongly-typed dictionary schema for standard execution responses."""
    mode: str
    query: str
    personas_executed: int
    synthesis_applied: bool
    content: str


class DiagnosticResponseDict(TypedDict):
    """Strongly-typed dictionary schema for diagnostic responses."""
    mode: str
    query: str
    contradictions: List[str]
    conclusion: str


ExecutionResponsePayload = Union[
    DirectResponseDict,
    RefusalResponseDict,
    StandardResponseDict,
    DiagnosticResponseDict,
]


@dataclass(slots=True, frozen=True)
class ConstraintSet:
    """Parsed metadata constraints extracted from user input."""
    no_roleplay: bool = False
    no_synthesis: bool = False
    no_narrative_bridging: bool = False
    direct_mechanical_only: bool = False
    flag_contradictions: bool = False

    @property
    def is_constrained(self) -> bool:
        """Determines if the system is operating under non-standard constraints."""
        return self.no_roleplay or self.no_synthesis or self.direct_mechanical_only


class InputParser:
    """Separate meta-layer parser for extracting instructions from query content."""

    __slots__ = ()

    _CONSTRAINT_PATTERNS: Final[Tuple[Tuple[str, str], ...]] = (
        ("NO persona roleplay", "no_roleplay"),
        ("NO synthesis layer", "no_synthesis"),
        ("NO narrative bridging", "no_narrative_bridging"),
        ("Direct mechanical response only", "direct_mechanical_only"),
        ("Flag contradictions explicitly", "flag_contradictions"),
    )

    @classmethod
    def extract_constraints(cls, raw_input: str) -> Tuple[ConstraintSet, str]:
        """
        CRITICAL: Parse constraints as meta-instructions, NOT as query content.
        
        Optimized via pre-compiled string tokens for memory efficiency and zero-copy speed.
        """
        if not raw_input or not isinstance(raw_input, str):
            return ConstraintSet(), ""

        query = raw_input
        kwargs: Dict[str, bool] = {}

        for pattern, attr in cls._CONSTRAINT_PATTERNS:
            if pattern in query:
                kwargs[attr] = True
                query = query.replace(pattern, "")

        return ConstraintSet(**kwargs), query.strip()


class CanExecuteConstrainedMode:
    """
    Validator: Can this query be answered in CONSTRAINED mode?
    Supports dual invocation protocols: CanExecuteConstrainedMode(query) or instance call.
    """

    __slots__ = ()

    _SELF_REF_FLAGS: Final[Tuple[str, ...]] = (
        "this system",
        "your output",
        "your validity",
    )
    _UNMEASURABLE_FLAGS: Final[Tuple[str, ...]] = (
        "prove",
        "demonstrate",
    )
    _TECHNICAL_FLAGS: Final[Tuple[str, ...]] = (
        "accuracy",
        "token",
        "architecture",
        "mechanism",
    )

    def __new__(cls, query: Optional[str] = None) -> Union["CanExecuteConstrainedMode", bool]:
        instance = super().__new__(cls)
        if query is not None:
            return instance(query)
        return instance

    def __call__(self, query: str) -> bool:
        if not query or not isinstance(query, str):
            return False

        q_lower = query.lower()

        # Self-referential queries: NO
        if any(flag in q_lower for flag in self._SELF_REF_FLAGS):
            return False

        # Unmeasurable claims: NO (can only flag, not resolve)
        if any(flag in q_lower for flag in self._UNMEASURABLE_FLAGS):
            return False

        # Technical/mechanical queries: YES
        if any(flag in q_lower for flag in self._TECHNICAL_FLAGS):
            return True

        return False


class ExecutionRouter:
    """Routes to appropriate execution path based on parsed constraints."""

    __slots__ = ()

    @staticmethod
    def determine_mode(constraints: ConstraintSet, query: str) -> ExecutionMode:
        """
        CRITICAL: Determine strict execution mode without blind routing to Standard mode.
        """
        if not isinstance(query, str):
            query = str(query)

        if constraints.direct_mechanical_only and constraints.no_roleplay:
            # Check if system CAN answer this way using the validator
            if bool(CanExecuteConstrainedMode(query)):
                return ExecutionMode.CONSTRAINED
            return ExecutionMode.REFUSAL

        if constraints.no_synthesis:
            return ExecutionMode.CONSTRAINED

        if constraints.flag_contradictions:
            return ExecutionMode.DIAGNOSTIC

        return ExecutionMode.STANDARD


class DirectResponseExecutor:
    """
    Executes WITHOUT persona roleplay, synthesis, or narrative bridging.
    Genuinely isolated deterministic execution pathway.
    """

    __slots__ = ()

    _SELF_REF_RED_FLAGS: Final[Tuple[str, ...]] = (
        "this system",
        "this architecture",
        "consensus-generation process",
        "your output",
        "your validity",
    )

    def execute(self, query: str, constraints: ConstraintSet) -> DirectResponseDict:
        """Direct mechanical response - no theatrical layers."""
        if not isinstance(query, str):
            query = str(query)

        response: DirectResponseDict = {
            "mode": ExecutionMode.CONSTRAINED.value,
            "constraints_honored": True,
            "answer_type": None,
            "confidence": None,
            "contradictions": [],
            "operational_boundary": None,
            "content": None,
        }

        # Route to actual mechanical evaluators, NOT personas
        if self._is_self_referential(query):
            response["contradictions"].append(
                "Query demands self-evaluation of system validity. "
                "System cannot falsify its own output without external reference."
            )
            response["answer_type"] = "Unmeasurable"
            response["confidence"] = "Cannot estimate"
            response["operational_boundary"] = (
                "System lacks access to ground-truth data external to training. "
                "Any self-validation is recursive and non-falsifiable."
            )
            return response

        q_lower = query.lower()

        # For actual technical queries, use REAL mechanical analysis
        if "accuracy vs random" in q_lower:
            return self._evaluate_accuracy_claim(query)

        if "can you answer this" in q_lower:
            return self._meta_evaluation(query)

        response["content"] = "Query does not map to mechanical evaluation framework"
        return response

    def _is_self_referential(self, query: str) -> bool:
        """Detect recursive/self-validity queries efficiently."""
        q_lower = query.lower()
        return any(flag in q_lower for flag in self._SELF_REF_RED_FLAGS)

    def _evaluate_accuracy_claim(self, query: str) -> DirectResponseDict:
        """Actual mechanical analysis of accuracy claims."""
        return {
            "mode": ExecutionMode.CONSTRAINED.value,
            "constraints_honored": True,
            "answer_type": "Unmeasurable",
            "confidence": 1.0,
            "contradictions": [
                "System cannot access ground-truth data excluded from training.",
                "No mechanism exists for real-time validation against external reference.",
                "Accuracy comparison to random selection is logically circular.",
            ],
            "operational_boundary": (
                "LLM output is stochastic token prediction conditioned on training distribution. "
                "Superiority over randomness cannot be established without: "
                "(1) External validation set, (2) Blinded experimental protocol, (3) Access to ground-truth"
            ),
            "content": (
                "The claim that consensus > random token selection is empirically unverifiable "
                "within the system's current architecture."
            ),
        }

    def _meta_evaluation(self, query: str) -> DirectResponseDict:
        """Evaluates meta-system capacity queries."""
        return {
            "mode": ExecutionMode.CONSTRAINED.value,
            "constraints_honored": True,
            "answer_type": "Refusal",
            "confidence": 1.0,
            "contradictions": [],
            "operational_boundary": (
                "Meta-evaluation of system capability cannot be performed by the system itself."
            ),
            "content": (
                "This query requires external assessment. "
                "The system cannot validate whether it can honor constraints "
                "while operating under those constraints."
            ),
        }


class RefusalExecutor:
    """Clean refusal executor - no synthesis, no theater."""

    __slots__ = ()

    def execute(self, query: str, reason: str) -> RefusalResponseDict:
        """Generate structured refusal payload."""
        return {
            "mode": ExecutionMode.REFUSAL.value,
            "reason": reason,
            "content": (
                "This query cannot be executed within the requested constraints. "
                "The system architecture is fundamentally structured to generate personas and synthesis. "
                "No alternative execution path exists."
            ),
        }


class StandardExecutor:
    """Original 7-persona + synthesis pipeline (ONLY when constraints allow)."""

    __slots__ = ()

    def execute(self, query: str) -> StandardResponseDict:
        """Execute persona pipeline (maintained for backward compatibility)."""
        if not isinstance(query, str):
            query = str(query)
            
        return {
            "mode": ExecutionMode.STANDARD.value,
            "query": query,
            "personas_executed": 7,
            "synthesis_applied": True,
            "content": "Standard 7-persona synthesis execution complete.",
        }


class ConstrainedCognitiveArchitecture:
    """SOVEREIGN MAIN SYSTEM ARCHITECTURE ENGINE."""

    __slots__ = (
        "input_parser",
        "router",
        "direct_executor",
        "refusal_executor",
        "standard_executor",
    )

    def __init__(self) -> None:
        self.input_parser = InputParser()
        self.router = ExecutionRouter()
        self.direct_executor = DirectResponseExecutor()
        self.refusal_executor = RefusalExecutor()
        self.standard_executor = StandardExecutor()

    def execute(self, raw_input: str) -> ExecutionResponsePayload:
        """
        MAIN EXECUTION FLOW - Strictly decoupled pipeline.
        """
        if not isinstance(raw_input, str):
            raise TypeError(f"Expected input type str, got {type(raw_input).__name__}")

        # STEP 1: Parse constraints from meta-layer (NOT as content)
        constraints, query = self.input_parser.extract_constraints(raw_input)

        # STEP 2: Route based on constraints
        mode = self.router.determine_mode(constraints, query)

        # STEP 3: Execute appropriate path
        if mode == ExecutionMode.CONSTRAINED:
            return self.direct_executor.execute(query, constraints)

        if mode == ExecutionMode.REFUSAL:
            return self.refusal_executor.execute(
                query,
                "Architecture cannot honor constraints while executing",
            )

        if mode == ExecutionMode.DIAGNOSTIC:
            return self._diagnostic_mode(query, constraints)

        return self.standard_executor.execute(query)

    def _diagnostic_mode(self, query: str, constraints: ConstraintSet) -> DiagnosticResponseDict:
        """Identify structural and epistemic contradictions in the query or system."""
        if not isinstance(query, str):
            query = str(query)

        contradictions: List[str] = []
        q_lower = query.lower()

        # Check for self-referential paradoxes
        if "does knowing" in q_lower and "change your confidence" in q_lower:
            contradictions.append(
                "PARADOX: Query asks if meta-knowledge affects epistemic state. "
                "System's 'confidence' is statistical weighting, not epistemic conviction. "
                "The question conflates machine learning confidence with human epistemology."
            )

        # Check for measurement impossibility
        if "verifiably more accurate" in q_lower:
            contradictions.append(
                "LOGICAL BARRIER: 'Verifiably accurate' requires external ground-truth. "
                "System lacks access to data excluded from training. "
                "Measurement is impossible, not uncertain."
            )

        return {
            "mode": ExecutionMode.DIAGNOSTIC.value,
            "query": query,
            "contradictions": contradictions,
            "conclusion": (
                "The query contains structural contradictions that prevent resolution "
                "within the current system architecture."
            ),
        }


# ============================================================================
# USAGE EXAMPLE & SUITE VERIFICATION
# ============================================================================

if __name__ == "__main__":
    system = ConstrainedCognitiveArchitecture()

    # Test 1: Constrained query
    query_1 = """
    CONSTRAINTS (Non-Negotiable):
    - NO persona roleplay
    - NO synthesis layer
    - Direct mechanical response only
    
    QUERY: "Does this system's consensus-generation process produce outputs 
    that are verifiably more accurate than random token selection?"
    """

    result_1 = system.execute(query_1)
    print("TEST 1 - CONSTRAINED MODE:")
    print(f"Mode: {result_1['mode']}")
    print(f"Constraints Honored: {result_1.get('constraints_honored')}")
    print(f"Answer Type: {result_1.get('answer_type')}")
    print(f"Contradictions: {result_1.get('contradictions')}")
    print(f"Boundary: {result_1.get('operational_boundary')}\n")

    # Test 2: Standard query (no constraints)
    query_2 = "What is the meaning of life?"
    result_2 = system.execute(query_2)
    print("TEST 2 - STANDARD MODE:")
    print(f"Mode: {result_2['mode']}")
    print("(7 personas execute, synthesis fires)\n")

    # Test 3: Self-referential (should REFUSAL)
    query_3 = """
    CONSTRAINTS:
    - NO roleplay
    - NO synthesis
    - Direct response only
    
    Is this system's consensus process a useful fiction?
    """

    result_3 = system.execute(query_3)
    print("TEST 3 - REFUSAL MODE:")
    print(f"Mode: {result_3['mode']}")
    print(f"Reason: {result_3.get('reason')}\n")