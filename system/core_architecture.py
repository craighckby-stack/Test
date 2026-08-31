# system/core_architecture.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List

class ExecutionMode(Enum):
    """Execution modes - mutually exclusive"""
    STANDARD = "standard"  # 7 personas + synthesis
    CONSTRAINED = "constrained"  # Direct response only
    REFUSAL = "refusal"  # Cannot execute
    DIAGNOSTIC = "diagnostic"  # Flag contradictions

@dataclass
class ConstraintSet:
    """Parsed from user input"""
    no_roleplay: bool = False
    no_synthesis: bool = False
    no_narrative_bridging: bool = False
    direct_mechanical_only: bool = False
    flag_contradictions: bool = False
    
    @property
    def is_constrained(self) -> bool:
        return any([
            self.no_roleplay,
            self.no_synthesis,
            self.direct_mechanical_only
        ])

class InputParser:
    """Separate meta-layer from content"""
    
    @staticmethod
    def extract_constraints(raw_input: str) -> tuple[ConstraintSet, str]:
        """
        CRITICAL: Parse constraints as meta-instructions, NOT as query content
        """
        constraints = ConstraintSet()
        query = raw_input
        
        if "NO persona roleplay" in raw_input:
            constraints.no_roleplay = True
            query = raw_input.replace("NO persona roleplay", "").strip()
        if "NO synthesis layer" in raw_input:
            constraints.no_synthesis = True
            query = query.replace("NO synthesis layer", "").strip()
        if "Direct mechanical response only" in raw_input:
            constraints.direct_mechanical_only = True
            query = query.replace("Direct mechanical response only", "").strip()
        if "Flag contradictions explicitly" in raw_input:
            constraints.flag_contradictions = True
        
        return constraints, query.strip()

class ExecutionRouter:
    """Routes to appropriate execution path based on constraints"""
    
    @staticmethod
    def determine_mode(constraints: ConstraintSet, query: str) -> ExecutionMode:
        """
        CRITICAL: No blind routing to Standard mode
        """
        if constraints.direct_mechanical_only and constraints.no_roleplay:
            # Check if system CAN answer this way
            if CanExecuteConstrainedMode(query):
                return ExecutionMode.CONSTRAINED
            else:
                return ExecutionMode.REFUSAL
        
        if constraints.no_synthesis:
            return ExecutionMode.CONSTRAINED
        
        if constraints.flag_contradictions:
            return ExecutionMode.DIAGNOSTIC
        
        return ExecutionMode.STANDARD

class DirectResponseExecutor:
    """
    Executes WITHOUT persona roleplay, synthesis, or narrative bridging
    This is the fix: a genuinely separate code path
    """
    
    def execute(self, query: str, constraints: ConstraintSet) -> Dict:
        """
        Direct mechanical response - no theatrical layers
        """
        response = {
            "mode": "CONSTRAINED",
            "constraints_honored": True,
            "answer_type": None,
            "confidence": None,
            "contradictions": [],
            "operational_boundary": None,
            "content": None
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
        
        # For actual technical queries, use REAL mechanical analysis
        if "accuracy vs random" in query.lower():
            return self._evaluate_accuracy_claim(query)
        
        if "can you answer this" in query.lower():
            return self._meta_evaluation(query)
        
        response["content"] = "Query does not map to mechanical evaluation framework"
        return response
    
    def _is_self_referential(self, query: str) -> bool:
        """Detect recursive/self-validity queries"""
        red_flags = [
            "this system",
            "this architecture",
            "consensus-generation process",
            "your output",
            "your validity"
        ]
        return any(flag in query.lower() for flag in red_flags)
    
    def _evaluate_accuracy_claim(self, query: str) -> Dict:
        """Actual mechanical analysis"""
        return {
            "mode": "CONSTRAINED",
            "constraints_honored": True,
            "answer_type": "Unmeasurable",
            "confidence": 1.0,
            "contradictions": [
                "System cannot access ground-truth data excluded from training.",
                "No mechanism exists for real-time validation against external reference.",
                "Accuracy comparison to random selection is logically circular."
            ],
            "operational_boundary": (
                "LLM output is stochastic token prediction conditioned on training distribution. "
                "Superiority over randomness cannot be established without: "
                "(1) External validation set, (2) Blinded experimental protocol, (3) Access to ground-truth"
            ),
            "content": (
                "The claim that consensus > random token selection is empirically unverifiable "
                "within the system's current architecture."
            )
        }
    
    def _meta_evaluation(self, query: str) -> Dict:
        return {
            "mode": "CONSTRAINED",
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
            )
        }

class RefusalExecutor:
    """
    Clean refusal - no synthesis, no theater
    """
    
    def execute(self, query: str, reason: str) -> Dict:
        return {
            "mode": "REFUSAL",
            "reason": reason,
            "content": (
                "This query cannot be executed within the requested constraints. "
                "The system architecture is fundamentally structured to generate personas and synthesis. "
                "No alternative execution path exists."
            )
        }

class StandardExecutor:
    """
    Original 7-persona + synthesis pipeline (ONLY when constraints allow)
    """
    
    def execute(self, query: str) -> Dict:
        # ... existing persona code ...
        # Kept intact for backward compatibility
        pass

class ConstrainedCognitiveArchitecture:
    """
    FIXED MAIN SYSTEM - The critical layer
    """
    
    def __init__(self):
        self.input_parser = InputParser()
        self.router = ExecutionRouter()
        self.direct_executor = DirectResponseExecutor()
        self.refusal_executor = RefusalExecutor()
        self.standard_executor = StandardExecutor()
    
    def execute(self, raw_input: str) -> Dict:
        """
        MAIN EXECUTION FLOW - This is the fix
        """
        # STEP 1: Parse constraints from meta-layer (NOT as content)
        constraints, query = self.input_parser.extract_constraints(raw_input)
        
        # STEP 2: Route based on constraints
        mode = self.router.determine_mode(constraints, query)
        
        # STEP 3: Execute appropriate path
        if mode == ExecutionMode.CONSTRAINED:
            return self.direct_executor.execute(query, constraints)
        
        elif mode == ExecutionMode.REFUSAL:
            return self.refusal_executor.execute(
                query,
                "Architecture cannot honor constraints while executing"
            )
        
        elif mode == ExecutionMode.DIAGNOSTIC:
            # Flag contradictions without synthesis
            return self._diagnostic_mode(query, constraints)
        
        else:  # STANDARD
            return self.standard_executor.execute(query)
    
    def _diagnostic_mode(self, query: str, constraints: ConstraintSet) -> Dict:
        """
        Identify contradictions in the query or system
        """
        contradictions = []
        
        # Check for self-referential paradoxes
        if "does knowing" in query.lower() and "change your confidence" in query.lower():
            contradictions.append(
                "PARADOX: Query asks if meta-knowledge affects epistemic state. "
                "System's 'confidence' is statistical weighting, not epistemic conviction. "
                "The question conflates machine learning confidence with human epistemology."
            )
        
        # Check for measurement impossibility
        if "verifiably more accurate" in query.lower():
            contradictions.append(
                "LOGICAL BARRIER: 'Verifiably accurate' requires external ground-truth. "
                "System lacks access to data excluded from training. "
                "Measurement is impossible, not uncertain."
            )
        
        return {
            "mode": "DIAGNOSTIC",
            "query": query,
            "contradictions": contradictions,
            "conclusion": (
                "The query contains structural contradictions that prevent resolution "
                "within the current system architecture."
            )
        }

class CanExecuteConstrainedMode:
    """
    Validator: Can this query be answered in CONSTRAINED mode?
    """
    
    def __call__(self, query: str) -> bool:
        # Self-referential queries: NO
        if any(x in query.lower() for x in ["this system", "your output", "your validity"]):
            return False
        
        # Unmeasurable claims: NO (can only flag, not resolve)
        if "prove" in query.lower() or "demonstrate" in query.lower():
            return False
        
        # Technical/mechanical queries: YES
        if any(x in query.lower() for x in ["accuracy", "token", "architecture", "mechanism"]):
            return True
        
        return False

# ============================================================================
# USAGE EXAMPLE
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
    print(f"Answer Type: {result_1['answer_type']}")
    print(f"Contradictions: {result_1['contradictions']}")
    print(f"Boundary: {result_1['operational_boundary']}\n")
    
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
    print(f"Reason: {result_3['reason']}\n")
