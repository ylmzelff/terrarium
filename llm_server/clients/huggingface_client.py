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


# Global model cache for sharing models across multiple clients (saves VRAM!)
_MODEL_CACHE = {}


class HuggingFaceClient(AbstractClient):
    """
    Client that loads and runs Hugging Face models locally using transformers library.
    
    No API keys required - models run directly on your hardware.
    Models are shared across all clients with the same model_name to save VRAM.
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
        
        # Use shared pipeline from cache (saves VRAM!)
        self._pipeline = None
        
        logger.info(f"Initialized HuggingFaceClient with model: {model_name} (shared across agents)")

    def _ensure_model_loaded(self):
        """Lazy load the model and tokenizer. Uses global cache to share models across agents."""
        # Check if already loaded from cache
        if self._pipeline is not None:
            return
        
        # Check global cache first (VRAM saver!)
        cache_key = f"{self.model_name}_{self.device}"
        if cache_key in _MODEL_CACHE:
            self._pipeline = _MODEL_CACHE[cache_key]
            logger.info(f"✅ Reusing cached model {self.model_name} (saves VRAM!)")
            return
        
        try:
            from transformers import pipeline
            import torch
            
            logger.info(f"Loading Hugging Face model: {self.model_name} (first time, will be cached)")
            
            # Check if CUDA is available
            device_map = self.device
            if device_map == "auto" and not torch.cuda.is_available():
                device_map = "cpu"
                logger.warning("CUDA not available, using CPU. This will be slower.")
            
            # Create text generation pipeline with memory optimization
            pipeline_kwargs = {
                "model": self.model_name,
                "device_map": device_map,
                "trust_remote_code": self.trust_remote_code,
                "max_memory": self.max_memory,
            }
            
            # Use FP16 on GPU for VRAM efficiency (2x memory save!)
            if torch.cuda.is_available():
                import transformers
                pipeline_kwargs["model_kwargs"] = {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                }
            
            self._pipeline = pipeline("text-generation", **pipeline_kwargs)
            
            # Cache the pipeline for other agents to reuse
            _MODEL_CACHE[cache_key] = self._pipeline
            
            logger.info(f"✅ Model {self.model_name} loaded and cached successfully on {device_map}")
            logger.info(f"💾 All agents will share this model instance (VRAM efficient)")
            
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

    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]] | None) -> str:
        """Format tools as a structured string for the model."""
        if not tools:
            return ""
        
        tool_descriptions = []
        for tool in tools:
            if tool.get("type") != "function":
                continue
            func = tool.get("function", {})
            name = func.get("name", "")
            description = func.get("description", "")
            parameters = func.get("parameters", {})
            
            tool_desc = f"## {name}\n{description}\n"
            if parameters:
                tool_desc += f"Parameters: {json.dumps(parameters)}\n"
            tool_descriptions.append(tool_desc)
        
        if tool_descriptions:
            return "\n# Available Tools:\n" + "\n".join(tool_descriptions) + "\n"
        return ""

    def _format_messages_for_model(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> str:
        """
        Format chat messages into a single prompt string with tool calling support.
        
        For chat models, we use the chat template if available.
        """
        self._ensure_model_loaded()
        
        # Add tool definitions to system message if tools are provided
        if tools:
            tool_info = self._format_tools_for_prompt(tools)
            # Find system message and append tool info
            system_found = False
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = msg["content"] + "\n" + tool_info + "\n" + (
                        "To call a tool, respond with JSON in this exact format:\n"
                        '{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}\n'
                        "You can call multiple tools at once."
                    )
                    system_found = True
                    break
            
            # If no system message, add one
            if not system_found:
                messages.insert(0, {
                    "role": "system",
                    "content": tool_info + "\n" + (
                        "To call a tool, respond with JSON in this exact format:\n"
                        '{"tool_calls": [{"name": "tool_name", "arguments": {...}}]}\n'
                        "You can call multiple tools at once."
                    )
                })
        
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
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                prompt_parts.append(f"Tool Result ({tool_name}): {content}")
        
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
            params: Generation parameters (max_tokens, temperature, tools, etc.)
            
        Returns:
            Tuple of (response_dict, response_string)
        """
        self._ensure_model_loaded()
        
        # Extract tools if provided
        tools = params.get("tools", [])
        
        # Format messages into prompt (with tools if available)
        prompt = self._format_messages_for_model(input, tools)
        
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
                pad_token_id=self._pipeline.tokenizer.eos_token_id,  # Prevent warnings
            )
            
            # Extract generated text
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].get("generated_text", "")
            else:
                generated_text = ""
            
            # Try to parse tool calls from the generated text
            tool_calls = self._parse_tool_calls(generated_text)
            
            # Build response in OpenAI-like format for compatibility
            message = {
                "role": "assistant",
                "content": generated_text if not tool_calls else None,
            }
            
            if tool_calls:
                message["tool_calls"] = tool_calls
                message["content"] = None  # When tool calls present, content is None
            
            response_dict = {
                "choices": [
                    {
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
                "model": self.model_name,
                "generated_text": generated_text,
            }
            
            return response_dict, generated_text
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from generated text.
        
        Looks for JSON objects with tool_calls in the format:
        {"tool_calls": [{"name": "tool_name", "arguments": {...}}]}
        """
        import re
        
        tool_calls = []
        
        # Try to find JSON blocks
        json_pattern = r'\{[^{}]*"tool_calls"[^{}]*\[.*?\].*?\}'
        matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                    for i, call in enumerate(data["tool_calls"]):
                        tool_name = call.get("name", "")
                        arguments = call.get("arguments", {})
                        
                        # Convert to OpenAI format
                        tool_calls.append({
                            "id": f"call_{i}_{hash(tool_name) % 10000}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments
                            }
                        })
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.debug(f"Failed to parse tool call from match: {e}")
                continue
        
        # Alternative: Look for function calls in more casual format
        if not tool_calls:
            # Pattern like: call_function(name="tool_name", args={"key": "value"})
            func_pattern = r'(?:call_function|function_call|tool)\s*\(\s*name\s*=\s*["\']([^"\']+)["\']\s*,\s*(?:args|arguments)\s*=\s*(\{[^}]+\})\s*\)'
            matches = re.finditer(func_pattern, text, re.IGNORECASE)
            
            for i, match in enumerate(matches):
                try:
                    tool_name = match.group(1)
                    args_str = match.group(2)
                    arguments = json.loads(args_str)
                    
                    tool_calls.append({
                        "id": f"call_{i}_{hash(tool_name) % 10000}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments)
                        }
                    })
                except (json.JSONDecodeError, IndexError) as e:
                    logger.debug(f"Failed to parse casual function call: {e}")
                    continue
        
        return tool_calls

    async def process_tool_calls(
        self,
        response: Dict[str, Any],
        context: List[Dict[str, Any]],
        execute_tool_callback: Any,
    ) -> Tuple[int, List[Dict[str, Any]], List[str]]:
        """
        Process tool calls from the model response.
        
        Now supports parsing tool calls from generated text!
        """
        choices = response.get("choices", [])
        if not choices:
            return 0, context, []

        message = choices[0].get("message", {})
        context.append(message)
        
        tool_calls = message.get("tool_calls", [])
        tool_calls_executed = 0
        step_tools: List[str] = []
        
        if not tool_calls:
            # No tool calls found
            logger.debug("No tool calls detected in model response")
            return 0, context, []
        
        logger.info(f"Detected {len(tool_calls)} tool call(s) in model response")
        
        for call in tool_calls:
            function_block = call.get("function") or {}
            tool_name = function_block.get("name", "unknown_tool")
            arguments_raw = function_block.get("arguments") or "{}"
            
            try:
                if isinstance(arguments_raw, str):
                    arguments = json.loads(arguments_raw)
                else:
                    arguments = arguments_raw
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool arguments: {e}, using empty dict")
                arguments = {}
            
            step_tools.append(f"{tool_name} -- {json.dumps(arguments)}")
            logger.info(f"Executing tool: {tool_name} with args: {arguments}")
            
            try:
                result = await execute_tool_callback(tool_name, arguments)
                
                context.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": tool_name,
                        "content": json.dumps(result),
                    }
                )
                tool_calls_executed += 1
                logger.info(f"Tool {tool_name} executed successfully")
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                context.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "name": tool_name,
                        "content": json.dumps({"error": str(e)}),
                    }
                )

        return tool_calls_executed, context, step_tools
