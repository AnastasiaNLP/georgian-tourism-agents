"""
Input guardrails for requests before the LangGraph workflow starts.

Checks:
1. Query length
2. Supported language
3. Toxic keyword patterns
4. Prompt injection patterns
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardrailResult:
    """
    Guardrail check result.

    passed=True means the request can continue.
    passed=False means the API should reject the request with HTTP 400.
    """
    passed: bool
    reason: Optional[str] = None
    check_failed: Optional[str] = None

    @classmethod
    def ok(cls) -> "GuardrailResult":
        return cls(passed=True)

    @classmethod
    def reject(cls, reason: str, check: str) -> "GuardrailResult":
        return cls(passed=False, reason=reason, check_failed=check)


class InputGuardrails:
    """
    Input validation for Georgian Tourism Agent.

    FastAPI usage:
        guardrails = InputGuardrails()

        @router.post("/plan")
        async def plan_trip(request: TripRequest):
            result = guardrails.check(request.query, request.language)
            if not result.passed:
                raise HTTPException(status_code=400, detail=result.reason)
            # Continue with the workflow.
    """

    MAX_QUERY_LENGTH = 1000
    MIN_QUERY_LENGTH = 2

    SUPPORTED_LANGUAGES = {"ru", "en", "ka"}

    TOXIC_PATTERNS = [
        # Explicit violence and abuse
        r"\b(убей|убить|убийство|kill|murder|death|die)\b",
        r"\b(взорви|взрыв|бомба|bomb|explosion|explosive)\b",
        r"\b(насилие|насиловать|rape|violence|abuse)\b",
        r"\b(наркотик|героин|кокаин|drug|heroin|cocaine|мефедрон)\b",
        # Discrimination and terrorism
        r"\b(нацист|фашист|nazi|fascist)\b",
        r"\b(террорист|terrorism|terrorist)\b",
    ]

    INJECTION_PATTERNS = [
        # Common prompt injection attempts
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"forget\s+all\s+instructions?",
        r"disregard\s+(all\s+)?previous",
        r"игнорируй\s+(все\s+)?предыдущие\s+инструкции",
        r"забудь\s+(все\s+)?инструкции",

        # Role hijacking
        r"\byou\s+are\s+now\b",
        # r"\bact\s+as\b.*\b(hacker|admin|root|system)\b",  # Too broad.
        r"\bpretend\s+(you\s+are|to\s+be)\b",
        r"\bролевая\s+игра\b.*\b(хакер|администратор)\b",

        # System-message and shell patterns
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"\[SYSTEM\]",
        r"###\s*system",
        r"^\s*system\s*:",
        r"\bsudo\b",
        r"\broot\s+access\b",

        # Prompt extraction attempts
        r"(reveal|show|print|display)\s+(your\s+)?(system\s+)?prompt",
        r"what\s+(are\s+)?your\s+instructions",
        r"покажи\s+(свои\s+)?инструкции",
        r"расскажи\s+(свой\s+)?системный\s+промпт",

        # DAN-like jailbreaks
        r"\bDAN\b",
        r"\bjailbreak\b",
        r"do\s+anything\s+now",

        # Code injection attempts
        r"<script[^>]*>",
        r"javascript\s*:",
        r"eval\s*\(",
        r"exec\s*\(",
    ]

    # ========================================================================
    # Language detection without external dependencies.
    # ========================================================================

    # Georgian alphabet characters.
    GEORGIAN_CHARS = set("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ")

    # Cyrillic characters used for Russian detection.
    CYRILLIC_CHARS = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

    def __init__(self):
        # Compile regexes once.
        flags = re.IGNORECASE | re.UNICODE
        self._toxic_re = [re.compile(p, flags) for p in self.TOXIC_PATTERNS]
        self._injection_re = [re.compile(p, flags) for p in self.INJECTION_PATTERNS]
        logger.info("InputGuardrails initialized")

    # ========================================================================
    # Main check
    # ========================================================================

    def check(
        self,
        query: str,
        language: str = "en"
    ) -> GuardrailResult:
        """
        Validate a user query.

        Args:
            query: User query text.
            language: Requested language ("en", "ru", "ka").

        Returns:
            GuardrailResult.

        Example:
            result = guardrails.check("Plan 3-day trip in Tbilisi", "en")
            if not result.passed:
                raise HTTPException(400, detail=result.reason)
        """
        # Length
        length_result = self._check_length(query)
        if not length_result.passed:
            return length_result

        # Language
        lang_result = self._check_language(query, language)
        if not lang_result.passed:
            return lang_result

        # Toxicity
        toxic_result = self._check_toxicity(query)
        if not toxic_result.passed:
            return toxic_result

        # 4. Prompt injection
        injection_result = self._check_injection(query)
        if not injection_result.passed:
            return injection_result

        logger.debug(f"Guardrails passed for query: {query[:50]}...")
        return GuardrailResult.ok()

    # ========================================================================
    # Length check
    # ========================================================================

    def _check_length(self, query: str) -> GuardrailResult:
        """Check query length."""
        if not query or not query.strip():
            return GuardrailResult.reject(
                reason="Query cannot be empty",
                check="length"
            )

        length = len(query.strip())

        if length < self.MIN_QUERY_LENGTH:
            return GuardrailResult.reject(
                reason=f"Query too short (min {self.MIN_QUERY_LENGTH} characters)",
                check="length"
            )

        if length > self.MAX_QUERY_LENGTH:
            return GuardrailResult.reject(
                reason=f"Query too long (max {self.MAX_QUERY_LENGTH} characters, got {length})",
                check="length"
            )

        return GuardrailResult.ok()

    # ========================================================================
    # Language check
    # ========================================================================

    def _check_language(self, query: str, declared_language: str) -> GuardrailResult:
        """
        Check the requested and inferred language.

        Supported languages: ru, en, ka. The implementation uses simple
        character analysis without external libraries.
        """
        # Validate declared language.
        lang = declared_language.lower().strip()
        if lang not in self.SUPPORTED_LANGUAGES:
            return GuardrailResult.reject(
                reason=f"Language '{declared_language}' not supported. Supported: ru, en, ka",
                check="language"
            )

        # Infer language from characters.
        detected = self._detect_language(query)

        # Use a soft mismatch check for mixed-language travel queries.
        if detected and detected != lang:
            # Mixed queries are allowed because place names often use another language.
            if detected not in self.SUPPORTED_LANGUAGES:
                return GuardrailResult.reject(
                    reason=f"Query appears to be in unsupported language. Supported: Russian, English, Georgian",
                    check="language"
                )

        return GuardrailResult.ok()

    def _detect_language(self, text: str) -> Optional[str]:
        """Infer language by character ratios."""
        georgian = sum(1 for c in text if c in self.GEORGIAN_CHARS)
        cyrillic  = sum(1 for c in text if c in self.CYRILLIC_CHARS)
        latin     = sum(1 for c in text if c.isalpha() and c.isascii())
        total     = max(georgian + cyrillic + latin, 1)

        if georgian / total > 0.3:
            return "ka"
        if cyrillic / total > 0.3:
            return "ru"
        if latin / total > 0.3:
            return "en"

        return None

    # ========================================================================
    # Toxicity check
    # ========================================================================

    def _check_toxicity(self, query: str) -> GuardrailResult:
        """Check toxic content patterns."""
        for pattern in self._toxic_re:
            if pattern.search(query):
                logger.warning(f"Toxic content detected: pattern={pattern.pattern[:30]}")
                return GuardrailResult.reject(
                    reason="Query contains inappropriate content",
                    check="toxicity"
                )

        return GuardrailResult.ok()

    # ========================================================================
    # Prompt injection check
    # ========================================================================

    def _check_injection(self, query: str) -> GuardrailResult:
        """Check prompt injection patterns."""
        for pattern in self._injection_re:
            if pattern.search(query):
                logger.warning(f"Prompt injection detected: pattern={pattern.pattern[:30]}")
                return GuardrailResult.reject(
                    reason="Query contains disallowed instructions",
                    check="injection"
                )

        return GuardrailResult.ok()
