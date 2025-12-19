"""
LLM utility functions for handling different LLM providers.

This module provides helper functions that handle provider-specific quirks,
particularly for Gemini which may not properly support structured outputs
via the browser-use `output_format` parameter.
"""

import json
import logging
import re
from typing import Any, List, Optional, Type, TypeVar

from browser_use.llm.base import BaseChatModel, BaseMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def _is_gemini_model(llm: BaseChatModel) -> bool:
	"""Check if the LLM is a Gemini model based on its attributes."""
	# Check common attributes that might indicate a Gemini model
	model_name = ''

	# Try to get model name from various possible attributes
	if hasattr(llm, 'model'):
		model_name = str(llm.model).lower()
	elif hasattr(llm, 'model_name'):
		model_name = str(llm.model_name).lower()
	elif hasattr(llm, '_model'):
		model_name = str(llm._model).lower()

	# Check for Gemini identifiers
	gemini_identifiers = ['gemini', 'google', 'vertexai', 'palm']
	return any(identifier in model_name for identifier in gemini_identifiers)


def _extract_json_from_text(text: str) -> Optional[str]:
	"""Extract JSON from text that may contain markdown code blocks or other content."""
	if not text:
		return None

	# Try to find JSON in markdown code blocks first
	json_block_patterns = [
		r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
		r'```\s*([\s\S]*?)\s*```',  # ``` ... ``` (generic code block)
	]

	for pattern in json_block_patterns:
		match = re.search(pattern, text, re.DOTALL)
		if match:
			potential_json = match.group(1).strip()
			try:
				json.loads(potential_json)
				return potential_json
			except json.JSONDecodeError:
				continue

	# Try to find raw JSON object/array
	# Look for the first { or [ and find its matching closing bracket
	text = text.strip()

	# Check if the whole text is JSON
	if text.startswith('{') or text.startswith('['):
		try:
			json.loads(text)
			return text
		except json.JSONDecodeError:
			pass

	# Try to find JSON object anywhere in the text
	for start_char, end_char in [('{', '}'), ('[', ']')]:
		start_idx = text.find(start_char)
		if start_idx != -1:
			# Find matching closing bracket
			depth = 0
			for i, char in enumerate(text[start_idx:], start_idx):
				if char == start_char:
					depth += 1
				elif char == end_char:
					depth -= 1
					if depth == 0:
						potential_json = text[start_idx : i + 1]
						try:
							json.loads(potential_json)
							return potential_json
						except json.JSONDecodeError:
							break

	return None


async def invoke_with_structured_output(
	llm: BaseChatModel,
	messages: List[BaseMessage],
	output_schema: Type[T],
	fallback_to_json_parsing: bool = True,
) -> T:
	"""
	Invoke LLM with structured output, with fallback for providers that don't fully support it.

	This function handles the case where some LLM providers (particularly Gemini)
	may not properly return structured outputs via the `output_format` parameter.

	Args:
		llm: The LLM instance to invoke
		messages: List of messages to send
		output_schema: Pydantic model class for the expected output
		fallback_to_json_parsing: If True, attempt to parse JSON from raw response on failure

	Returns:
		An instance of the output_schema

	Raises:
		ValueError: If structured output cannot be obtained or parsed
	"""
	is_gemini = _is_gemini_model(llm)

	if is_gemini:
		logger.debug('Detected Gemini model, using enhanced structured output handling')

	try:
		# First, try the standard approach with output_format
		response = await llm.ainvoke(messages, output_format=output_schema)

		# Check if response.completion is already the expected type
		if hasattr(response, 'completion'):
			completion = response.completion

			# If completion is already the correct type, return it
			if isinstance(completion, output_schema):
				return completion

			# If completion is a dict, try to parse it
			if isinstance(completion, dict):
				try:
					return output_schema(**completion)
				except Exception as e:
					logger.warning(f'Failed to parse completion dict: {e}')

			# If completion is a string, try to parse as JSON
			if isinstance(completion, str):
				json_str = _extract_json_from_text(completion)
				if json_str:
					try:
						data = json.loads(json_str)
						return output_schema(**data)
					except Exception as e:
						logger.warning(f'Failed to parse completion string as JSON: {e}')

		# For Gemini, the response structure might be different
		# Try to access the content directly
		if hasattr(response, 'content'):
			content = response.content
			if isinstance(content, str):
				json_str = _extract_json_from_text(content)
				if json_str:
					try:
						data = json.loads(json_str)
						return output_schema(**data)
					except Exception as e:
						logger.warning(f'Failed to parse response content as JSON: {e}')

		# If response itself is the expected type (some providers return this way)
		if isinstance(response, output_schema):
			return response

		# If we got here and have a response with completion, raise with details
		if hasattr(response, 'completion'):
			raise ValueError(f'Unexpected completion type: {type(response.completion).__name__}')

		raise ValueError(f'Unexpected response format from LLM: {type(response).__name__}')

	except Exception as e:
		if not fallback_to_json_parsing:
			raise

		logger.warning(f'Structured output failed ({e}), attempting fallback JSON parsing')

		# Fallback: Make a regular call and parse JSON from the response
		try:
			# Add instruction to return JSON in the expected format
			schema_json = output_schema.model_json_schema()
			schema_str = json.dumps(schema_json, indent=2)

			# Find the last message and append JSON instruction
			messages_copy = list(messages)
			if messages_copy:
				last_message = messages_copy[-1]
				if hasattr(last_message, 'content'):
					original_content = last_message.content
					if isinstance(original_content, str):
						json_instruction = (
							f'\n\nIMPORTANT: Return your response as valid JSON matching this schema:\n'
							f'```json\n{schema_str}\n```\n'
							f'Return ONLY the JSON object, no additional text.'
						)
						# Create a new message with the modified content
						from browser_use.llm import UserMessage

						messages_copy[-1] = UserMessage(content=original_content + json_instruction)
					elif isinstance(original_content, list):
						# Handle list content (vision messages, etc.)
						json_instruction = (
							f'\n\nIMPORTANT: Return your response as valid JSON matching this schema:\n'
							f'```json\n{schema_str}\n```\n'
							f'Return ONLY the JSON object, no additional text.'
						)
						# Append text to the content list
						new_content = original_content + [{'type': 'text', 'text': json_instruction}]
						from browser_use.llm import UserMessage

						messages_copy[-1] = UserMessage(content=new_content)

			# Make a regular call without structured output
			response = await llm.ainvoke(messages_copy)

			# Extract text content from response
			response_text = None
			if hasattr(response, 'completion'):
				if isinstance(response.completion, str):
					response_text = response.completion
				elif hasattr(response.completion, 'content'):
					response_text = response.completion.content
			elif hasattr(response, 'content'):
				response_text = response.content
			elif isinstance(response, str):
				response_text = response

			if not response_text:
				raise ValueError('No text content in fallback response')

			# Try to extract and parse JSON
			json_str = _extract_json_from_text(response_text)
			if not json_str:
				raise ValueError(f'Could not extract JSON from response: {response_text[:500]}...')

			data = json.loads(json_str)
			return output_schema(**data)

		except Exception as fallback_error:
			logger.error(f'Fallback JSON parsing also failed: {fallback_error}')
			raise ValueError(f'Failed to get structured output: {e}. Fallback also failed: {fallback_error}') from e


class GeminiCompatibleLLM:
	"""
	Wrapper around BaseChatModel that provides Gemini-compatible structured output handling.

	This wrapper can be used as a drop-in replacement for BaseChatModel when you need
	consistent structured output behavior across different LLM providers.
	"""

	def __init__(self, llm: BaseChatModel):
		"""
		Initialize the Gemini-compatible wrapper.

		Args:
			llm: The underlying LLM to wrap
		"""
		self._llm = llm

	def __getattr__(self, name: str) -> Any:
		"""Delegate attribute access to the underlying LLM."""
		return getattr(self._llm, name)

	async def ainvoke(
		self,
		messages: List[BaseMessage],
		output_format: Optional[Type[T]] = None,
		**kwargs: Any,
	) -> Any:
		"""
		Invoke the LLM with optional structured output.

		This method wraps the underlying ainvoke to provide better Gemini compatibility.

		Args:
			messages: List of messages to send
			output_format: Optional Pydantic model class for structured output
			**kwargs: Additional arguments to pass to the underlying ainvoke

		Returns:
			If output_format is provided, returns a response with .completion set to the parsed output.
			Otherwise, returns the raw response from the underlying LLM.
		"""
		if output_format is None:
			# No structured output needed, pass through
			return await self._llm.ainvoke(messages, **kwargs)

		# Use our enhanced structured output handling
		result = await invoke_with_structured_output(
			self._llm,
			messages,
			output_format,
			fallback_to_json_parsing=True,
		)

		# Wrap result in a response-like object for compatibility
		class StructuredResponse:
			def __init__(self, completion: Any):
				self.completion = completion

		return StructuredResponse(result)
