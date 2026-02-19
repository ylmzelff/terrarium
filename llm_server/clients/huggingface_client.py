"""
Hugging Face Transformers client for local model inference without API keys.

This client uses the transformers library to load and run models directly
from Hugging Face Hub, completely free and without requiring any API keys.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from llm_server.clients.abstract_client import AbstractClient

logger = logging.getLogger(__name__)


class HuggingFaceClient(AbstractClient):
    """
    Client that loads and runs Hugging Face models locally using transformers library.
    
    No API keys required - models run directly on your hardware.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        trust_remote_code: bool = True,
        max_memory: Dict[int, str] | None = None,
    ):
        """
        Initialize Hugging Face client with a model.
        
        Args:
            model_name: Hugging Face model identifier (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
            device: Device to run on ("auto", "cuda", "cpu", or specific device like "cuda:0")
            trust_remote_code: Whether to trust remote code (required for some models)
            max_memory: Memory allocation dict for multi-GPU setups
        """
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.max_memory = max_memory
        
        # Lazy loading - only load when needed
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        
        logger.info(f"Initialized HuggingFaceClient with model: {model_name}")

    def _ensure_model_loaded(self):
        """Lazy load the model and tokenizer."""
        if self._pipeline is not None:
            return
        
        try:
            from transformers import pipeline
            import torch
            
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            
            # Check if CUDA is available
            device_map = self.device
            if device_map == "auto" and not torch.cuda.is_available():
                device_map = "cpu"
                logger.warning("CUDA not available, using CPU. This will be slower.")
            
            # Create text generation pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map=device_map,
                trust_remote_code=self.trust_remote_code,
                max_memory=self.max_memory,
            )
            
            logger.info(f"Model {self.model_name} loaded successfully on {device_map}")
            
        except ImportError as e:
            raise ImportError(
                "transformers and torch libraries are required for HuggingFaceClient. "
                "Install with: pip install transformers torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e

    @staticmethod
    def init_context(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
        """Initialize chat-style context."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_message_content(message: Dict[str, Any]) -> str:
        """Extract text content from response message."""
        if isinstance(message, dict):
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return "".join(parts)
        return ""

    @staticmethod
    def get_usage(response: Dict[str, Any], current_usage: Dict[str, int]) -> Dict[str, int]:
        """
        Update token usage statistics.
        
        Note: Local inference doesn't provide token counts from transformers,
        so we estimate them.
        """
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        if "generated_text" in response:
            text = response["generated_text"]
            estimated_tokens = len(text) // 4
            current_usage["completion_tokens"] += estimated_tokens
            current_usage["total_tokens"] += estimated_tokens
        
        return current_usage

    def _format_messages_for_model(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format chat messages into a single prompt string.
        
        For chat models, we use the chat template if available.
        """
        self._ensure_model_loaded()
        
        # Try to use the model's chat template if available
        if hasattr(self._pipeline, "tokenizer") and hasattr(self._pipeline.tokenizer, "apply_chat_template"):
            try:
                return self._pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, falling back to manual formatting")
        
        # Fallback: manual formatting
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def generate_response(
        self,
        input: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Generate a response using the local Hugging Face model.
        
        Args:
            input: List of message dictionaries (chat format)
            params: Generation parameters (max_tokens, temperature, etc.)
            
        Returns:
            Tuple of (response_dict, response_string)
        """
        self._ensure_model_loaded()
        
        # Format messages into prompt
        prompt = self._format_messages_for_model(input)
        
        # Extract generation parameters
        max_tokens = params.get("max_completion_tokens") or params.get("max_tokens", 256)
        temperature = params.get("temperature", 0.7)
        
        # Generate response
        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,  # Only return generated text, not the prompt
            )
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].get("generated_text", "")
            else:
                generated_text = ""
            
            # Build response in OpenAI-like format for compatibility
            response_dict = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "model": self.model_name,
                "generated_text": generated_text,
            }
            
            return response_dict, generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e

    async def process_tool_calls(
        self,
        response: Dict[str, Any],
        context: List[Dict[str, Any]],
        execute_tool_callback: Any,
    ) -> Tuple[int, List[Dict[str, Any]], List[str]]:
        """
        Process tool calls from the model response.
        
        Note: Basic open-source models may not support structured tool calling.
        This implementation looks for JSON-formatted tool calls in the text.
        """
        choices = response.get("choices", [])
        if not choices:
            return 0, context, []

        message = choices[0].get("message", {})
        context.append(message)
        
        # Try to parse tool calls from message content
        content = message.get("content", "")
        
        # Simple tool call detection: look for JSON blocks
        # More sophisticated models might have structured tool call format
        tool_calls_executed = 0
        step_tools: List[str] = []
        
        # For now, return no tool calls (basic models don't support this)
        # Advanced users can extend this for models with tool-calling capabilities
        logger.debug(f"Tool calling not supported by basic models. Response: {content[:100]}...")
        
        return tool_calls_executed, context, step_tools
