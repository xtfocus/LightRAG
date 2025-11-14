from collections.abc import Iterable
import os
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncAzureOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from openai.types.chat import ChatCompletionMessageParam

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    safe_unicode_decode,
    logger,
    verbose_debug,
    VERBOSE_DEBUG,
)

import numpy as np

# Try to import Langfuse for LLM observability (optional)
# Falls back to no instrumentation if not available
# Langfuse requires proper configuration to work correctly
LANGFUSE_ENABLED = False
get_langfuse_client = None

# Create a no-op decorator as default
def _no_op_observe(*args, **kwargs):
    """No-op decorator when Langfuse is not available or not configured"""
    def decorator(func):
        return func
    return decorator

observe = _no_op_observe

# Initialize Langfuse configuration with visible logging
langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
langfuse_host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Always log Langfuse configuration status at INFO level
logger.info(f"Langfuse configuration check: PUBLIC_KEY={'set' if langfuse_public_key else 'not set'}, SECRET_KEY={'set' if langfuse_secret_key else 'not set'}, HOST={langfuse_host}")

try:
    # Only enable Langfuse if both keys are configured
    if langfuse_public_key and langfuse_secret_key:
        try:
            from langfuse import observe as langfuse_observe, get_client

            observe = langfuse_observe
            get_langfuse_client = get_client
            LANGFUSE_ENABLED = True
            logger.info(f"Langfuse observability ENABLED for Azure OpenAI client (host: {langfuse_host})")
        except ImportError as e:
            logger.warning(f"Langfuse package not installed. Install with: pip install langfuse. Error: {e}")
            logger.info("Langfuse observability DISABLED for Azure OpenAI client (package not installed)")
    else:
        missing = []
        if not langfuse_public_key:
            missing.append("LANGFUSE_PUBLIC_KEY")
        if not langfuse_secret_key:
            missing.append("LANGFUSE_SECRET_KEY")
        logger.info(f"Langfuse observability DISABLED for Azure OpenAI client. Missing environment variables: {', '.join(missing)}")
        logger.debug(
            "Langfuse environment variables not configured, Azure OpenAI calls will not be traced"
        )
except Exception as e:
    logger.warning(f"Error initializing Langfuse: {e}")
    logger.info("Langfuse observability DISABLED for Azure OpenAI client")


@observe(name="azure-openai-complete", as_type="generation")
async def _azure_openai_complete_inner(
    model: str,
    prompt: str,
    messages: list[dict],
    base_url: str | None,
    deployment: str,
    api_key: str | None,
    api_version: str | None,
    timeout: int | None,
    kwargs: dict,
    trace_metadata: dict,
):
    """Inner function that performs the actual API call with Langfuse observability.
    
    This function is decorated with @observe and receives cleaned arguments
    (without hashing_kv) so Langfuse can properly capture input/output without
    exceeding response size limits.
    """
    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=base_url,
        azure_deployment=deployment,
        api_key=api_key,
        api_version=api_version,
        timeout=timeout,
    )

    try:
        if "response_format" in kwargs:
            logger.debug("Using beta chat.completions.parse endpoint")
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            logger.debug("Using standard chat.completions.create endpoint")
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        

        if hasattr(response, "__aiter__"):
            # Streaming response - create a wrapper that collects data
            final_chunk_usage = None
            collected_content = []

            async def stream_inner():
                nonlocal final_chunk_usage, collected_content
                try:
                    chunk_count = 0
                    async for chunk in response:
                        chunk_count += 1
                        # Check if this chunk has usage information (final chunk)
                        if hasattr(chunk, "usage") and chunk.usage:
                            final_chunk_usage = chunk.usage
                        # Check if choices exists and is not empty
                        if not hasattr(chunk, "choices") or not chunk.choices:
                            continue

                        # Check if delta exists
                        if not hasattr(chunk.choices[0], "delta"):
                            # This might be the final chunk, continue to check for usage
                            continue

                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content is None:
                            continue
                        if r"\u" in content:
                            content = safe_unicode_decode(content.encode("utf-8"))
                        collected_content.append(content)
                        yield content

                    # Update trace after stream completes
                    # Note: This executes after the generator is exhausted
                    if LANGFUSE_ENABLED and get_langfuse_client:
                        try:
                            langfuse = get_langfuse_client()
                            full_content = "".join(collected_content)
                            stream_metadata = {
                                **trace_metadata,
                                "streaming": True,
                            }
                            if final_chunk_usage:
                                stream_metadata["usage"] = {
                                    "prompt_tokens": getattr(final_chunk_usage, "prompt_tokens", 0),
                                    "completion_tokens": getattr(
                                        final_chunk_usage, "completion_tokens", 0
                                    ),
                                    "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
                                }
                            langfuse.update_current_trace(
                                input={"prompt": prompt, "messages": messages},
                                output={"content": full_content, "streaming": True},
                                metadata=stream_metadata,
                            )
                        except Exception as trace_error:
                            # Don't fail if trace update fails
                            logger.warning(f"Failed to update Langfuse trace for streaming: {trace_error}")
                except Exception as e:
                    # Ensure trace is updated with error before re-raising
                    if LANGFUSE_ENABLED and get_langfuse_client:
                        try:
                            langfuse = get_langfuse_client()
                            langfuse.update_current_trace(
                                input={"prompt": prompt, "messages": messages},
                                output={"error": str(e), "streaming": True},
                                metadata={**trace_metadata, "error": True, "streaming": True},
                            )
                        except Exception as trace_error:
                            logger.warning(f"Failed to update Langfuse trace with streaming error: {trace_error}")
                    raise
                finally:
                    # Ensure client is closed for streaming responses
                    try:
                        await openai_async_client.close()
                    except Exception as close_error:
                        logger.warning(f"Failed to close Azure OpenAI client: {close_error}")

            return stream_inner()
        else:
            # Non-streaming response
            try:
                # Validate response structure
                if (
                    not response
                    or not response.choices
                    or not hasattr(response.choices[0], "message")
                ):
                    logger.error("Invalid response from Azure OpenAI API")
                    raise ValueError("Invalid response from Azure OpenAI API")

                message = response.choices[0].message
                content = getattr(message, "content", None)

                # Validate content is not empty
                if content is None or (isinstance(content, str) and content.strip() == ""):
                    logger.error("Received empty content from Azure OpenAI API")
                    raise ValueError("Received empty content from Azure OpenAI API")

                # Apply Unicode decoding if needed
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))

                verbose_debug(f"Response: {content[:200]}..." if len(content) > 200 else f"Response: {content}")
                
                # Log token usage if available
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                    completion_tokens = getattr(response.usage, "completion_tokens", 0)
                    total_tokens = getattr(response.usage, "total_tokens", 0)
                    logger.info(f"Azure OpenAI token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

                # Update trace with token usage and metadata
                if LANGFUSE_ENABLED and get_langfuse_client:
                    try:
                        langfuse = get_langfuse_client()
                        trace_data = {
                            "input": {"prompt": prompt, "messages": messages},
                            "output": {"content": content},
                            "metadata": trace_metadata,
                        }

                        # Add token usage if available
                        if hasattr(response, "usage") and response.usage:
                            trace_data["metadata"]["usage"] = {
                                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                                "completion_tokens": getattr(
                                    response.usage, "completion_tokens", 0
                                ),
                                "total_tokens": getattr(response.usage, "total_tokens", 0),
                            }

                        langfuse.update_current_trace(**trace_data)
                    except Exception as trace_error:
                        # Don't fail the function if trace update fails
                        logger.warning(f"Failed to update Langfuse trace: {trace_error}")

                return content
            finally:
                # Ensure client is closed for non-streaming responses
                await openai_async_client.close()
    except Exception as e:
        logger.error(f"Azure OpenAI API Call Failed, Model: {model}, Deployment: {deployment}, Error: {e}")
        # Update trace with error information
        if LANGFUSE_ENABLED and get_langfuse_client:
            try:
                langfuse = get_langfuse_client()
                langfuse.update_current_trace(
                    input={"prompt": prompt, "messages": messages},
                    output={"error": str(e)},
                    metadata={**trace_metadata, "error": True},
                )
            except Exception as trace_error:
                # Don't fail if trace update fails
                logger.warning(f"Failed to update Langfuse trace with error: {trace_error}")
        # Ensure client is closed even on error
        try:
            await openai_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close Azure OpenAI client on error: {close_error}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt: str | None = None,
    history_messages: Iterable[ChatCompletionMessageParam] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    **kwargs,
):
    """
    Complete a prompt using Azure OpenAI's API.
    
    Note: This function removes large parameters (like hashing_kv) before creating
    Langfuse traces to avoid exceeding response size limits.
    """
    # Remove large parameters BEFORE creating any traces
    # This prevents Langfuse from capturing hashing_kv which can be huge
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    timeout = kwargs.pop("timeout", None)
    
    if enable_cot:
        logger.debug(
            "enable_cot=True is not supported for the Azure OpenAI API and will be ignored."
        )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or model or os.getenv("LLM_MODEL")
    base_url = (
        base_url or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BINDING_HOST")
    )
    api_key = (
        api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    )
    api_version = (
        api_version
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
    )

    # Prepare messages for trace metadata
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    # Prepare trace metadata
    trace_metadata = {
        "model": model,
        "deployment": deployment,
        "api_version": api_version,
        "base_url": base_url,
        "enable_cot": enable_cot,
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
        "stream": kwargs.get("stream", False),
    }

    # Add debug logging similar to openai.py
    logger.info(f"Azure OpenAI LLM call: Model={model}, Deployment={deployment}, API Version={api_version}")
    logger.debug("===== Entering Azure OpenAI LLM function =====")
    logger.debug(f"Model: {model}   Deployment: {deployment}")
    logger.debug(f"Base URL: {base_url}   API Version: {api_version}")
    logger.debug(f"Additional kwargs: {kwargs}")
    logger.debug(f"Num of history messages: {len(history_messages) if history_messages else 0}")
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to Azure OpenAI LLM =====")

    # Call the inner function which is decorated with @observe
    # This ensures hashing_kv is not captured by Langfuse (it's removed from kwargs above)
    return await _azure_openai_complete_inner(
        model=model,
        prompt=prompt,
        messages=messages,
        base_url=base_url,
        deployment=deployment,
        api_key=api_key,
        api_version=api_version,
        timeout=timeout,
        kwargs=kwargs,
        trace_metadata=trace_metadata,
    )


async def azure_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    logger.info("azure_openai_complete function called")
    logger.debug("===== azure_openai_complete called =====")
    kwargs.pop("keyword_extraction", None)
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    logger.debug(f"Using model from env: {model}")
    result = await azure_openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    logger.info("azure_openai_complete function completed")
    logger.debug("===== azure_openai_complete completed =====")
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def azure_openai_embed(
    texts: list[str],
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
) -> np.ndarray:
    deployment = (
        os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        or model
        or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    base_url = (
        base_url
        or os.getenv("AZURE_EMBEDDING_ENDPOINT")
        or os.getenv("EMBEDDING_BINDING_HOST")
    )
    api_key = (
        api_key
        or os.getenv("AZURE_EMBEDDING_API_KEY")
        or os.getenv("EMBEDDING_BINDING_API_KEY")
    )
    api_version = (
        api_version
        or os.getenv("AZURE_EMBEDDING_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
    )

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=base_url,
        azure_deployment=deployment,
        api_key=api_key,
        api_version=api_version,
    )

    try:
        response = await openai_async_client.embeddings.create(
            model=model, input=texts, encoding_format="float"
        )

        # Validate response structure
        if not response or not hasattr(response, "data") or not response.data:
            logger.error("Invalid response from Azure OpenAI Embedding API")
            raise ValueError("Invalid response from Azure OpenAI Embedding API")

        embeddings = np.array([dp.embedding for dp in response.data])

        embedding_dim = int(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0
        logger.info(f"Azure OpenAI embeddings generated: shape={embeddings.shape}, dimension={embedding_dim}")
        logger.debug(f"Generated embeddings with shape: {embeddings.shape}")
        logger.debug(f"Embedding dimension: {embedding_dim}")
        
        # Log token usage if available
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            total_tokens = getattr(response.usage, "total_tokens", 0)
            logger.info(f"Azure OpenAI embedding token usage: prompt={prompt_tokens}, total={total_tokens}")
        
        logger.debug("===== Azure OpenAI Embedding function completed =====")
        return embeddings
    except Exception as e:
        logger.error(f"Azure OpenAI Embedding API Call Failed, Model: {model}, Deployment: {deployment}, Error: {e}")
        # Update trace with error information
        if LANGFUSE_ENABLED and get_langfuse_client:
            try:
                langfuse = get_langfuse_client()
                langfuse.update_current_trace(
                    input={"text_count": len(texts), "total_text_length": sum(len(text) for text in texts)},
                    output={"error": str(e)},
                    metadata={**trace_metadata, "error": True},
                )
            except Exception as trace_error:
                # Don't fail if trace update fails
                logger.warning(f"Failed to update Langfuse trace with error: {trace_error}")
        raise
    finally:
        # Ensure client is closed
        try:
            await openai_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close Azure OpenAI client: {close_error}")
