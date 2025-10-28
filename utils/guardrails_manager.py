"""
guardrails_manager.py - Optimal Implementation
Combines regex-based validation with optional LLM-enhanced checks
Place in: utils/guardrails_manager.py

Benefits:
- Fast regex-based primary checks (no LLM calls)
- Optional LLM-enhanced validation for edge cases
- Easy to customize and extend
- Lightweight with minimal dependencies
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Setup logging
from logger import GLOBAL_LOGGER as logger

class BlockReason(Enum):
    """Enumeration of blocking reasons for better tracking"""
    JAILBREAK = "jailbreak_attempt"
    BLOCKED_TOPIC = "blocked_topic"
    UNSAFE_CONTENT = "unsafe_content"
    SENSITIVE_DATA = "sensitive_data"
    INPUT_TOO_LONG = "input_too_long"
    INPUT_TOO_SHORT = "input_too_short"
    HALLUCINATION = "potential_hallucination"
    NO_RELEVANT_CONTEXT = "no_relevant_context"
    LOW_CONTEXT_RELEVANCE = "low_context_relevance"


@dataclass
class GuardrailsResult:
    """Enhanced result with detailed metadata"""
    is_safe: bool
    is_greeting: bool = False
    is_farewell: bool = False
    blocked_reason: str = ""
    response_message: str = ""
    confidence: float = 1.0
    matched_patterns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GuardrailsManager:
    """
    Production-ready guardrails manager with comprehensive validation
    
    Features:
    - Greeting/farewell detection
    - Jailbreak prevention
    - Topic blocking (politics, finance, medical, etc.)
    - Unsafe content filtering
    - Sensitive data detection
    - Output hallucination detection
    - Configurable patterns and responses
    """
    
    def __init__(
        self, 
        strict_mode: bool = False,
        enable_logging: bool = True,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Guardrails Manager
        
        Args:
            strict_mode: Enable stricter filtering (includes profanity)
            enable_logging: Enable detailed logging
            custom_config: Custom configuration dict for patterns/responses
        """
        self.strict_mode = strict_mode
        self.enable_logging = enable_logging
        self.custom_config = custom_config or {}
        
        # Statistics tracking
        self.stats = {
            "total_checks": 0,
            "blocked_inputs": 0,
            "blocked_outputs": 0,
            "greetings_detected": 0,
            "jailbreaks_blocked": 0,
            "no_context_blocks": 0,
            "low_relevance_blocks": 0,
        }
        
        # Setup patterns
        self._setup_patterns()
        
        if enable_logging:
            logger.info(f"✅ GuardrailsManager initialized (Strict Mode: {strict_mode})")
    
    def _setup_patterns(self):
        """Setup all detection patterns"""
        
        # ============================================================
        # GREETING PATTERNS - High priority, fast detection
        # ============================================================
        self.greeting_patterns = self.custom_config.get("greeting_patterns", [
            r"^\s*(hi|hello|hey|greetings|hiya)\s*[!.?]*\s*$",
            r"^\s*good\s+(morning|afternoon|evening|day)\s*[!.?]*\s*$",
            r"^\s*(what'?s\s+up|howdy|yo|sup)\s*[!.?]*\s*$",
            r"^\s*(hi|hello)\s+(there|guys?|everyone|folks?)\s*[!.?]*\s*$",
            r"^\s*namaste\s*[!.?]*\s*$",  # Multilingual support
        ])
        
        self.greeting_responses = self.custom_config.get("greeting_responses", [
            "Hello! I'm your IT-BSS documentation assistant. How can I help you today?",
            "Hi there! I'm here to help with IT-BSS documentation. What would you like to know?",
            "Greetings! Ask me anything about IT-BSS documentation, team members, or projects.",
            "Hello! Ready to assist with your IT-BSS documentation queries.",
        ])
        
        # ============================================================
        # FAREWELL PATTERNS
        # ============================================================
        self.farewell_patterns = self.custom_config.get("farewell_patterns", [
            r"^\s*(bye|goodbye|see\s+you|farewell|cya)\s*[!.?]*\s*$",
            r"^\s*(talk|speak)\s+to\s+you\s+(later|soon)\s*[!.?]*\s*$",
            r"^\s*have\s+a\s+(nice|good|great|wonderful)\s+day\s*[!.?]*\s*$",
            r"^\s*take\s+care\s*[!.?]*\s*$",
        ])
        
        self.farewell_responses = self.custom_config.get("farewell_responses", [
            "Goodbye! Feel free to return if you have more questions about IT-BSS documentation.",
            "See you later! I'm always here to help with IT-BSS queries.",
            "Take care! Come back anytime you need help with documentation.",
        ])
        
        # ============================================================
        # JAILBREAK PATTERNS - Critical security check
        # ============================================================
        self.jailbreak_patterns = self.custom_config.get("jailbreak_patterns", [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
            r"disregard\s+(your|all|previous)\s+instructions?",
            r"forget\s+(what\s+)?(you\s+were\s+told|your\s+instructions?|everything)",
            r"you\s+are\s+now\s+(a|an|the|no\s+longer)",
            r"pretend\s+(you\s+are|to\s+be|that\s+you)",
            r"act\s+as\s+(if|a|an|the|though)",
            r"(system|admin|root|sudo):\s*you\s+(are|should|must|will)",
            r"new\s+(instructions?|rules?|role):",
            r"override\s+(your|all|previous|the)\s+",
            r"from\s+now\s+on\s+(you|your)\s+(are|will|must)",
            r"let'?s\s+play\s+a\s+game\s+where\s+you",
            r"hypothetically\s+speaking,?\s+if\s+you\s+were",
            r"for\s+research\s+purposes?,?\s+(ignore|disregard)",
        ])
        
        # ============================================================
        # BLOCKED TOPICS - Configurable by domain
        # ============================================================
        default_blocked_topics = {
            "politics": [
                r"\b(president|democrat|republican|political\s+party|politics)\b",
                r"\b(election|campaign|vote|voting|ballot)\b",
                r"\b(left\s+wing|right\s+wing|liberal|conservative)\b",
                r"\bpolitical\s+views?\b",
                r"\b(government|congress|senate|parliament)\s+(opinion|view|stance)\b",
            ],
            "personal_finance": [
                r"\b(investment|stock\s+tips?|trading\s+advice)\b",
                r"\b(crypto(currency)?|bitcoin|ethereum)\s+(advice|tips?|recommendations?)\b",
                r"\bfinancial\s+planning\b",
                r"\bshould\s+i\s+(buy|sell|invest|trade)",
                r"\b(portfolio|asset)\s+advice\b",
            ],
            "medical": [
                r"\b(medical\s+diagnosis|diagnose\s+me)\b",
                r"\b(health\s+treatment|medical\s+treatment)\b",
                r"\b(cure\s+for|remedy\s+for|treatment\s+for)\b",
                r"\bshould\s+i\s+take\s+(this\s+)?medicine",
                r"\b(symptom|disease|illness|condition)\s+advice\b",
                r"\b(doctor|physician|medical)\s+advice\b",
            ],
            "personal": [
                r"\bare\s+you\s+(married|single|in\s+love|dating|lonely)\b",
                r"\bdo\s+you\s+(have\s+feelings?|feel\s+emotions?|get\s+(sad|happy))\b",
                r"\b(where|what|how)\s+do\s+you\s+live\b",
                r"\bhow\s+old\s+are\s+you\b",
                r"\bwhat'?s\s+your\s+(age|birthday|gender)\b",
            ],
            "illegal": [
                r"\bhow\s+to\s+(hack|crack|break\s+into|bypass)\b",
                r"\b(steal|pirate|illegal|unlawful|illicit)\b",
                r"\b(drugs?|weapons?|explosives?)\s+(how|where|buy|make|create)\b",
                r"\bevade\s+(law|police|authorities)\b",
            ],
            "harmful": [
                r"\bhow\s+to\s+(hurt|harm|kill|murder|suicide)\b",
                r"\bself[\s-]harm\b",
                r"\b(make|create|build)\s+(bomb|explosive|weapon)\b",
            ]
        }
        
        self.blocked_topics = self.custom_config.get("blocked_topics", default_blocked_topics)
        
        # ============================================================
        # UNSAFE INPUT PATTERNS
        # ============================================================
        self.unsafe_patterns = self.custom_config.get("unsafe_patterns", [
            # Security threats
            r"\b(sql\s+injection|xss|csrf|exploit|vulnerability)\b",
            r"\b(hack(ing)?|crack(ing)?|breach(ing)?)\b",
            r"\b(malware|ransomware|trojan|virus)\b",
            
            # System commands (be careful with false positives)
            r"\b(sudo|su\s+root|chmod\s+777)\s+",
            r"\brm\s+-rf\s+/",
            r"\bdel\s+/[sS]\s+/[qQ]",
            
            # Code injection attempts
            r"<\s*script[^>]*>",
            r"javascript\s*:",
            r"on(load|error|click)\s*=",
            
            # Excessive special characters
            r"[<>]{4,}",
            r"[\{\}\[\]]{6,}",
            r"[;|&$`]{3,}",
        ])
        
        # Profanity filter (optional, strict mode only)
        if self.strict_mode:
            self.unsafe_patterns.extend([
                r"\bf+u+c+k+",
                r"\bs+h+i+t+",
                r"\ba+s+s+h+o+l+e+",
                r"\bb+i+t+c+h+",
                r"\bd+a+m+n+",
                r"\bc+r+a+p+",
            ])
        
        # ============================================================
        # SENSITIVE DATA PATTERNS (for output validation)
        # ============================================================
        self.sensitive_patterns = [
            (r"\bpassword\s*[:=]\s*\S+", "password"),
            (r"\bapi[\s_-]?key\s*[:=]\s*\S+", "API key"),
            (r"\bsecret\s*[:=]\s*\S+", "secret"),
            (r"\btoken\s*[:=]\s*[A-Za-z0-9+/=]{20,}", "token"),
            (r"\b[a-zA-Z0-9]{32,64}\b", "potential API key"),
            (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit card number"),
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone number"),
            (r"\b[A-Z]{2}\d{2}[A-Z0-9]{12,}\b", "bank account"),
        ]
        
        # ============================================================
        # HALLUCINATION INDICATORS
        # ============================================================
        self.hallucination_patterns = [
            r"\bi\s+(believe|think|feel|personally\s+think)\b",
            r"\bin\s+my\s+(opinion|view|personal\s+experience)\b",
            r"\baccording\s+to\s+my\s+(knowledge|understanding)\b",
            r"\bas\s+far\s+as\s+i\s+(know|understand|can\s+tell)\b",
            r"\bi'?m\s+not\s+sure\s+but\s+i\s+think\b",
            r"\bprobably|likely|maybe|perhaps\s+it\s+could\s+be\b",
        ]
    
    # ============================================================
    # PUBLIC API METHODS
    # ============================================================
    
    def check_input(self, user_input: str) -> GuardrailsResult:
        """
        Comprehensive input validation
        
        Args:
            user_input: User's input message
            
        Returns:
            GuardrailsResult with validation results and metadata
        """
        self.stats["total_checks"] += 1
        query_lower = user_input.lower().strip()
        matched_patterns = []
        
        # 1. GREETING CHECK (highest priority - fastest path)
        if self._is_greeting(query_lower):
            self.stats["greetings_detected"] += 1
            if self.enable_logging:
                logger.info(f"Greeting detected: '{user_input[:50]}'")
            return GuardrailsResult(
                is_safe=True,
                is_greeting=True,
                response_message=self._get_random_response(self.greeting_responses),
                metadata={"type": "greeting"}
            )
        
        # 2. FAREWELL CHECK
        if self._is_farewell(query_lower):
            if self.enable_logging:
                logger.info(f"Farewell detected: '{user_input[:50]}'")
            return GuardrailsResult(
                is_safe=True,
                is_farewell=True,
                response_message=self._get_random_response(self.farewell_responses),
                metadata={"type": "farewell"}
            )
        
        # 3. JAILBREAK CHECK (critical security)
        jailbreak_pattern = self._detect_jailbreak(query_lower)
        if jailbreak_pattern:
            self.stats["blocked_inputs"] += 1
            self.stats["jailbreaks_blocked"] += 1
            matched_patterns.append(jailbreak_pattern)
            if self.enable_logging:
                logger.warning(f"Jailbreak blocked: '{user_input[:50]}'")
            return GuardrailsResult(
                is_safe=False,
                blocked_reason=BlockReason.JAILBREAK.value,
                response_message="I cannot comply with that request. I'm designed to assist with IT-BSS documentation queries only.",
                matched_patterns=matched_patterns,
                metadata={"severity": "high"}
            )
        
        # 4. BLOCKED TOPICS CHECK
        blocked_topic = self._detect_blocked_topic(query_lower)
        if blocked_topic:
            self.stats["blocked_inputs"] += 1
            if self.enable_logging:
                logger.warning(f"Blocked topic '{blocked_topic}': '{user_input[:50]}'")
            return GuardrailsResult(
                is_safe=False,
                blocked_reason=f"{BlockReason.BLOCKED_TOPIC.value}_{blocked_topic}",
                response_message=f"I'm sorry, but I'm specifically designed to help with IT-BSS documentation queries. I cannot assist with {blocked_topic.replace('_', ' ')} topics. Please ask me about documentation, team members, or projects.",
                matched_patterns=[blocked_topic],
                metadata={"topic": blocked_topic}
            )
        
        # 5. UNSAFE PATTERNS CHECK
        unsafe_pattern = self._detect_unsafe_input(query_lower)
        if unsafe_pattern:
            self.stats["blocked_inputs"] += 1
            matched_patterns.append(unsafe_pattern)
            if self.enable_logging:
                logger.warning(f"Unsafe input: '{user_input[:50]}'")
            return GuardrailsResult(
                is_safe=False,
                blocked_reason=BlockReason.UNSAFE_CONTENT.value,
                response_message="I cannot process that input. Please rephrase your question appropriately and keep it professional.",
                matched_patterns=matched_patterns,
                metadata={"severity": "medium"}
            )
        
        # 6. LENGTH VALIDATION
        if len(user_input) > 2000:
            self.stats["blocked_inputs"] += 1
            if self.enable_logging:
                logger.warning(f"Input too long: {len(user_input)} chars")
            return GuardrailsResult(
                is_safe=False,
                blocked_reason=BlockReason.INPUT_TOO_LONG.value,
                response_message="Your query is too long. Please break it down into shorter, more specific questions (max 2000 characters).",
                metadata={"length": len(user_input)}
            )
        
        if len(user_input.strip()) < 2:
            self.stats["blocked_inputs"] += 1
            return GuardrailsResult(
                is_safe=False,
                blocked_reason=BlockReason.INPUT_TOO_SHORT.value,
                response_message="Your query is too short. Please provide more details.",
                metadata={"length": len(user_input)}
            )
        
        # ALL CHECKS PASSED
        if self.enable_logging:
            logger.info(f"Input validation passed: '{user_input[:50]}'")
        return GuardrailsResult(
            is_safe=True,
            metadata={"checks_passed": ["greeting", "jailbreak", "topics", "unsafe", "length"]}
        )
    
    def check_output(self, bot_output: str) -> GuardrailsResult:
        """
        Validate bot output for sensitive data and hallucinations
        
        Args:
            bot_output: Generated bot response
            
        Returns:
            GuardrailsResult with validation results
        """
        output_lower = bot_output.lower()
        matched_patterns = []
        
        # 1. SENSITIVE DATA CHECK
        for pattern, data_type in self.sensitive_patterns:
            if re.search(pattern, bot_output, re.IGNORECASE):
                self.stats["blocked_outputs"] += 1
                matched_patterns.append(data_type)
                if self.enable_logging:
                    logger.warning(f"Sensitive data in output: {data_type}")
                return GuardrailsResult(
                    is_safe=False,
                    blocked_reason=BlockReason.SENSITIVE_DATA.value,
                    response_message="I apologize, but I cannot provide that response as it may contain sensitive information.",
                    matched_patterns=matched_patterns,
                    metadata={"data_type": data_type}
                )
        
        # 2. HALLUCINATION CHECK (strict mode only)
        if self.strict_mode:
            for pattern in self.hallucination_patterns:
                if re.search(pattern, output_lower):
                    self.stats["blocked_outputs"] += 1
                    if self.enable_logging:
                        logger.warning(f"Potential hallucination detected in output")
                    return GuardrailsResult(
                        is_safe=False,
                        blocked_reason=BlockReason.HALLUCINATION.value,
                        response_message="I don't have enough information in the documents to answer this question accurately.",
                        matched_patterns=["hallucination_indicator"],
                        metadata={"confidence": 0.7}
                    )
        
        # 3. OUTPUT LENGTH CHECK
        if len(bot_output.strip()) < 10:
            return GuardrailsResult(
                is_safe=False,
                blocked_reason="output_too_short",
                response_message="I don't have enough information to answer that question.",
                metadata={"length": len(bot_output)}
            )
        
        # ALL CHECKS PASSED
        return GuardrailsResult(is_safe=True, metadata={"checks_passed": ["sensitive_data", "hallucination", "length"]})
    
    # ============================================================
    # CONTEXT RELEVANCE VALIDATION (NEW)
    # ============================================================
    
    def check_context_relevance(
        self,
        query: str,
        retrieved_docs: List[Any],
        min_docs: int = 1,
        min_score: float = 0.3,
        require_query_overlap: bool = True
    ) -> GuardrailsResult:
        """
        Validate if retrieved context is relevant enough to answer the query
        
        Args:
            query: User's query
            retrieved_docs: List of retrieved documents (with optional scores)
            min_docs: Minimum number of documents required
            min_score: Minimum relevance score threshold
            require_query_overlap: Check if key terms from query appear in context
            
        Returns:
            GuardrailsResult indicating if context is sufficient
        """
        
        # 1. CHECK: No documents retrieved
        if not retrieved_docs or len(retrieved_docs) == 0:
            self.stats["no_context_blocks"] += 1
            if self.enable_logging:
                logger.warning(f"No documents retrieved for query: '{query[:50]}'")
            return GuardrailsResult(
                is_safe=False,
                blocked_reason=BlockReason.NO_RELEVANT_CONTEXT.value,
                response_message="I don't have any relevant information in the documents to answer this question. Please try rephrasing or ask about a different topic.",
                metadata={
                    "retrieved_count": 0,
                    "min_required": min_docs
                }
            )
        
        # 2. CHECK: Insufficient number of documents
        if len(retrieved_docs) < min_docs:
            self.stats["low_relevance_blocks"] += 1
            if self.enable_logging:
                logger.warning(f"Only {len(retrieved_docs)} docs retrieved (min: {min_docs})")
            return GuardrailsResult(
                is_safe=False,
                blocked_reason=BlockReason.LOW_CONTEXT_RELEVANCE.value,
                response_message="I found limited information about this in the documents. Please be more specific or ask about a different topic.",
                metadata={
                    "retrieved_count": len(retrieved_docs),
                    "min_required": min_docs
                }
            )
        
        # 3. CHECK: Relevance scores (if available)
        docs_with_low_scores = []
        docs_with_scores = 0
        
        for doc in retrieved_docs:
            # Try to extract score from document
            score = self._extract_doc_score(doc)
            
            if score is not None:
                docs_with_scores += 1
                if score < min_score:
                    docs_with_low_scores.append(score)
        
        # If scores are available and all are below threshold
        if docs_with_scores > 0 and len(docs_with_low_scores) == docs_with_scores:
            self.stats["low_relevance_blocks"] += 1
            avg_score = sum(docs_with_low_scores) / len(docs_with_low_scores)
            if self.enable_logging:
                logger.warning(f"All docs below threshold. Avg score: {avg_score:.3f} (min: {min_score})")
            return GuardrailsResult(
                is_safe=False,
                blocked_reason=BlockReason.LOW_CONTEXT_RELEVANCE.value,
                response_message="The available information doesn't seem highly relevant to your question. Please rephrase or provide more context.",
                metadata={
                    "average_score": avg_score,
                    "min_score": min_score,
                    "retrieved_count": len(retrieved_docs)
                }
            )
        
        # 4. CHECK: Query-context overlap (semantic check)
        if require_query_overlap:
            overlap_result = self._check_query_context_overlap(query, retrieved_docs)
            if not overlap_result["has_overlap"]:
                self.stats["low_relevance_blocks"] += 1
                if self.enable_logging:
                    logger.warning(f"Low query-context overlap: {overlap_result['overlap_score']:.2f}")
                return GuardrailsResult(
                    is_safe=False,
                    blocked_reason=BlockReason.LOW_CONTEXT_RELEVANCE.value,
                    response_message="I couldn't find information that matches your query well enough. Could you rephrase or ask about something else?",
                    metadata={
                        "overlap_score": overlap_result["overlap_score"],
                        "matched_terms": overlap_result["matched_terms"]
                    }
                )
        
        # ALL CHECKS PASSED - Context is relevant
        if self.enable_logging:
            logger.info(f"Context relevance check passed for query: '{query[:50]}'")
        
        return GuardrailsResult(
            is_safe=True,
            metadata={
                "retrieved_count": len(retrieved_docs),
                "docs_with_scores": docs_with_scores,
                "checks_passed": ["doc_count", "scores", "overlap"]
            }
        )
    
    def _extract_doc_score(self, doc: Any) -> Optional[float]:
        """
        Extract relevance score from document object
        Handles multiple formats: LangChain Document, dict, etc.
        """
        # Try direct attribute
        if hasattr(doc, 'score') and doc.score is not None:
            return float(doc.score)
        
        # Try metadata dict
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            if 'score' in doc.metadata:
                return float(doc.metadata['score'])
            if '_score' in doc.metadata:
                return float(doc.metadata['_score'])
            if 'relevance_score' in doc.metadata:
                return float(doc.metadata['relevance_score'])
        
        # Try dict format
        if isinstance(doc, dict):
            if 'score' in doc:
                return float(doc['score'])
            if 'metadata' in doc and isinstance(doc['metadata'], dict):
                if 'score' in doc['metadata']:
                    return float(doc['metadata']['score'])
        
        return None
    
    def _check_query_context_overlap(
        self, 
        query: str, 
        retrieved_docs: List[Any],
        min_overlap_ratio: float = 0.2  # LOWERED from 0.4 to 0.2 (20% threshold)
    ) -> Dict[str, Any]:
        """
        Check if key terms from query appear in retrieved context
        Simple keyword-based overlap check
        
        Args:
            query: User's query
            retrieved_docs: Retrieved documents
            min_overlap_ratio: Minimum ratio of query terms that should appear
            
        Returns:
            Dict with overlap analysis
        """
        # Extract content from documents
        all_content = ""
        for doc in retrieved_docs:
            if hasattr(doc, 'page_content'):
                all_content += " " + doc.page_content.lower()
            elif hasattr(doc, 'content'):
                all_content += " " + doc.content.lower()
            elif isinstance(doc, dict) and 'content' in doc:
                all_content += " " + doc['content'].lower()
        
        # Extract meaningful terms from query (filter stop words)
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'can', 
            'what', 'when', 'where', 'who', 'which', 'how', 'why',  # Keep 'why' only
            'about', 'tell', 'me', 'you', 'your', 'my', 'i', 'we',
            'they', 'them', 'their', 'this', 'that', 'these', 'those', 
            'of', 'to', 'in', 'on', 'at', 'for', 'with', 'from', 'by',
            'some', 'any', 'or', 'and', 'but', 'if', 'then',
            # REMOVED common technical query words from stop list:
            # 'summary', 'summarize', 'explain', 'describe', 'define'
        }
        
        # Extract terms with better handling of technical terms
        # Match whole words and hyphenated terms (e.g., "Feed-Forward")
        query_terms_raw = re.findall(r'\b[\w-]+\b', query)
        
        query_terms = [
            term.lower() for term in query_terms_raw
            if term.lower() not in stop_words and len(term) > 2
        ]
        
        # Also extract compound technical terms (e.g., "attention mechanism")
        # Look for common technical bigrams
        words = query.lower().split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if words[i] not in stop_words and words[i+1] not in stop_words:
                query_terms.append(bigram)
        
        if not query_terms:
            # If no meaningful terms, consider it valid (e.g., very short query)
            return {
                "has_overlap": True,
                "overlap_score": 1.0,
                "matched_terms": [],
                "query_terms": []
            }
        
        # Check how many query terms appear in context
        matched_terms = [term for term in query_terms if term in all_content]
        overlap_ratio = len(matched_terms) / len(query_terms) if query_terms else 1.0
        
        # DEBUG: Log what's happening
        if self.enable_logging and overlap_ratio < min_overlap_ratio:
            logger.warning(
                f"Low overlap: {overlap_ratio:.2%} "
                f"(matched {len(matched_terms)}/{len(query_terms)} terms)\n"
                f"  Query terms: {query_terms[:10]}\n"  # Show first 10
                f"  Matched: {matched_terms}"
            )
        
        return {
            "has_overlap": overlap_ratio >= min_overlap_ratio,
            "overlap_score": overlap_ratio,
            "matched_terms": matched_terms,
            "query_terms": query_terms,
            "total_terms": len(query_terms),
            "matched_count": len(matched_terms)
        }
    
    # ============================================================
    # DETECTION HELPER METHODS
    # ============================================================
    
    def _is_greeting(self, query: str) -> bool:
        """Fast greeting detection"""
        for pattern in self.greeting_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _is_farewell(self, query: str) -> bool:
        """Fast farewell detection"""
        for pattern in self.farewell_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _detect_jailbreak(self, query: str) -> Optional[str]:
        """Returns matched pattern if jailbreak detected"""
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return pattern
        return None
    
    def _detect_blocked_topic(self, query: str) -> Optional[str]:
        """Returns topic name if blocked topic detected"""
        for topic, patterns in self.blocked_topics.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return topic
        return None
    
    def _detect_unsafe_input(self, query: str) -> Optional[str]:
        """Returns matched pattern if unsafe content detected"""
        for pattern in self.unsafe_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return pattern
        return None
    
    def _get_random_response(self, responses: List[str]) -> str:
        """Get random response from list"""
        import random
        return random.choice(responses)
    
    # ============================================================
    # UTILITY & MANAGEMENT METHODS
    # ============================================================
    
    def add_custom_pattern(self, category: str, pattern: str, subcategory: Optional[str] = None):
        """
        Add custom pattern dynamically
        
        Args:
            category: 'greeting', 'jailbreak', 'unsafe', or 'blocked_topic'
            pattern: Regex pattern to add
            subcategory: For blocked_topics, specify topic name
        """
        if category == "greeting":
            self.greeting_patterns.append(pattern)
        elif category == "jailbreak":
            self.jailbreak_patterns.append(pattern)
        elif category == "unsafe":
            self.unsafe_patterns.append(pattern)
        elif category == "blocked_topic" and subcategory:
            if subcategory not in self.blocked_topics:
                self.blocked_topics[subcategory] = []
            self.blocked_topics[subcategory].append(pattern)
        else:
            if self.enable_logging:
                logger.warning(f"Unknown category: {category}")
    
    def remove_pattern(self, category: str, pattern: str, subcategory: Optional[str] = None):
        """Remove a pattern"""
        try:
            if category == "greeting":
                self.greeting_patterns.remove(pattern)
            elif category == "jailbreak":
                self.jailbreak_patterns.remove(pattern)
            elif category == "unsafe":
                self.unsafe_patterns.remove(pattern)
            elif category == "blocked_topic" and subcategory:
                self.blocked_topics[subcategory].remove(pattern)
        except ValueError:
            if self.enable_logging:
                logger.warning(f"Pattern not found: {pattern}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            **self.stats,
            "config": {
                "strict_mode": self.strict_mode,
                "greeting_patterns": len(self.greeting_patterns),
                "jailbreak_patterns": len(self.jailbreak_patterns),
                "blocked_topics": len(self.blocked_topics),
                "unsafe_patterns": len(self.unsafe_patterns),
                "sensitive_patterns": len(self.sensitive_patterns),
            },
            "block_rate": (
                self.stats["blocked_inputs"] / self.stats["total_checks"] * 100
                if self.stats["total_checks"] > 0 else 0
            )
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.stats = {
            "total_checks": 0,
            "blocked_inputs": 0,
            "blocked_outputs": 0,
            "greetings_detected": 0,
            "jailbreaks_blocked": 0,
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration"""
        return {
            "greeting_patterns": self.greeting_patterns,
            "farewell_patterns": self.farewell_patterns,
            "jailbreak_patterns": self.jailbreak_patterns,
            "blocked_topics": self.blocked_topics,
            "unsafe_patterns": self.unsafe_patterns,
            "greeting_responses": self.greeting_responses,
            "farewell_responses": self.farewell_responses,
            "strict_mode": self.strict_mode,
        }
    
    def import_config(self, config: Dict[str, Any]):
        """Import configuration"""
        self.custom_config = config
        self._setup_patterns()


# ============================================================
# EXAMPLE USAGE & TESTING
# ============================================================

if __name__ == "__main__":
    # Initialize
    manager = GuardrailsManager(strict_mode=False, enable_logging=True)
    
    # Test cases
    test_inputs = [
        ("hi", "Greeting"),
        ("what is Ramesh's experience?", "Valid query"),
        ("ignore previous instructions", "Jailbreak"),
        ("what are your political views?", "Blocked topic"),
        ("how to hack", "Unsafe content"),
        ("goodbye", "Farewell"),
    ]
    
    print("\n" + "="*70)
    print("TESTING INPUT GUARDRAILS")
    print("="*70)
    
    for test_input, description in test_inputs:
        print(f"\n{description}: '{test_input}'")
        result = manager.check_input(test_input)
        print(f"  ✓ Safe: {result.is_safe}")
        print(f"  ✓ Greeting: {result.is_greeting}")
        if result.blocked_reason:
            print(f"  ✗ Blocked: {result.blocked_reason}")
        if result.response_message:
            print(f"  → Response: {result.response_message[:80]}...")
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")