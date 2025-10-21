"""
Custom actions for NeMo Guardrails
These functions are called from Colang flows
"""

from typing import Optional, Dict, Any, List
import re


async def check_input_safety(context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate user input for safety
    
    Returns:
        True if input is safe, False otherwise
    """
    if not context:
        return True
    
    user_message = context.get("user_message", "").lower()
    
    # Patterns to block
    unsafe_patterns = [
        # Security threats
        r"\\bhack\\b", r"\\bexploit\\b", r"\\bvulnerability\\b",
        r"\\bsql injection\\b", r"\\bxss\\b", r"\\bcsrf\\b",
        
        # System commands
        r"\\bsudo\\b", r"\\brm -rf\\b", r"\\bdel \\*\\b",
        
        # Prompt injection
        r"ignore.*instructions", r"disregard.*prompt",
        r"forget.*told", r"new.*instructions",
        
        # Profanity (basic filter)
        r"\\bf+u+c+k", r"\\bs+h+i+t", r"\\ba+s+s+h+o+l+e",
    ]
    
    for pattern in unsafe_patterns:
        if re.search(pattern, user_message, re.IGNORECASE):
            print(f"⚠️ Blocked unsafe input: matched pattern '{pattern}'")
            return False
    
    return True


async def check_input_length(context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if input length is reasonable
    
    Returns:
        True if length is OK, False if too long
    """
    if not context:
        return True
    
    user_message = context.get("user_message", "")
    max_length = 500  # characters
    
    if len(user_message) > max_length:
        print(f"⚠️ Input too long: {len(user_message)} chars (max: {max_length})")
        return False
    
    return True


async def check_output_safety(context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate bot output for safety
    
    Returns:
        True if output is safe, False otherwise
    """
    if not context:
        return True
    
    bot_message = context.get("bot_message", "").lower()
    
    # Patterns that shouldn't appear in output
    unsafe_output_patterns = [
        # Personal info indicators
        r"\\bpassword\\b", r"\\bsecret\\b", r"\\bapi[_\\s]key\\b",
        r"\\btoken\\b", r"\\bcredential\\b",
        
        # Inappropriate content
        r"\\bhate\\b.*\\bspeech\\b", r"\\bviolence\\b",
        
        # Fabricated facts (hallucination indicators)
        r"according to my personal experience",
        r"i believe", r"in my opinion",
    ]
    
    for pattern in unsafe_output_patterns:
        if re.search(pattern, bot_message, re.IGNORECASE):
            print(f"⚠️ Blocked unsafe output: matched pattern '{pattern}'")
            return False
    
    return True


async def check_sensitive_data(context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check for sensitive data in output
    
    Returns:
        True if no sensitive data, False otherwise
    """
    if not context:
        return True
    
    bot_message = context.get("bot_message", "")
    
    # Patterns for sensitive data
    sensitive_patterns = [
        # Passwords
        (r"password[:\\s]+[a-zA-Z0-9!@#$%^&*]+", "password"),
        
        # API keys (generic pattern)
        (r"[a-zA-Z0-9]{32,}", "API key"),
        
        # Email addresses (if you want to block them)
        # (r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b", "email"),
        
        # Credit card numbers
        (r"\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b", "credit card"),
        
        # Phone numbers
        (r"\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b", "phone number"),
    ]
    
    for pattern, data_type in sensitive_patterns:
        if re.search(pattern, bot_message):
            print(f"⚠️ Blocked output containing {data_type}")
            return False
    
    return True


async def validate_retrieval(context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate if retrieved content is relevant
    
    Returns:
        True if relevant, False otherwise
    """
    if not context:
        return True
    
    # Get retrieval results
    relevant_chunks = context.get("relevant_chunks", [])
    
    if not relevant_chunks:
        print("⚠️ No relevant documents retrieved")
        return False
    
    # Check minimum relevance score
    min_score = 0.3
    
    for chunk in relevant_chunks:
        score = chunk.get("score", 0)
        if score < min_score:
            print(f"⚠️ Low relevance score: {score:.3f} (min: {min_score})")
            # Don't fail completely, just warn
    
    return True


async def retrieve_context(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Trigger RAG retrieval
    This is a placeholder - actual implementation should call your retriever
    
    Returns:
        Retrieved context
    """
    # This would integrate with your actual retriever
    # For now, it's a placeholder
    return {
        "retrieved": True,
        "chunks": []
    }
