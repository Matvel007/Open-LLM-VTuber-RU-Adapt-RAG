from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    safe_websocket_send,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message, get_history
from ..service_context import ServiceContext
from ..rag.memory_processor import process_memory_background

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()
    full_response = ""  # Initialize full_response here

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Cleanup old dialogue memory (async, non-blocking)
        if (
            context.dialogue_memory
            and context.system_config
            and context.system_config.rag_config
        ):
            try:
                days = getattr(
                    context.system_config.rag_config,
                    "memory_cleanup_days",
                    30,
                )
                asyncio.create_task(
                    asyncio.to_thread(
                        context.dialogue_memory.delete_older_than_days,
                        days=days,
                    )
                )
            except Exception:
                pass

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )

        # RAG: retrieve relevant context if enabled
        rag_context: list[str] = []
        if context.rag_engine and input_text.strip():
            try:
                rag_config = (
                    context.system_config.rag_config
                    if context.system_config and context.system_config.rag_config
                    else None
                )
                n_results = rag_config.n_results if rag_config else 5
                rag_context = context.rag_engine.query(
                    input_text.strip(), n_results=n_results
                )
                if rag_context:
                    logger.debug(f"RAG retrieved {len(rag_context)} chunks")
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")

        # Dialogue memory: user profile + similar past messages
        memory_context: list[str] = []
        if context.dialogue_memory and context.history_uid and input_text.strip():
            try:
                rag_config = (
                    context.system_config.rag_config
                    if context.system_config and context.system_config.rag_config
                    else None
                )
                mem_n = getattr(rag_config, "memory_n_results", 5) if rag_config else 5
                profile = context.dialogue_memory.get_user_profile(
                    context.history_uid, context.character_config.conf_uid
                )
                if profile:
                    memory_context.append(
                        f"Профиль пользователя (ОБЯЗАТЕЛЬНО используй при ответе):\n{profile}"
                    )
                similar = context.dialogue_memory.query(
                    input_text.strip(),
                    n_results=mem_n,
                    history_uid=context.history_uid,
                    conf_uid=context.character_config.conf_uid,
                    roles=["fact", "summary"],
                )
                if similar:
                    lines = []
                    for content, _, meta in similar:
                        role = meta.get("role", "")
                        if role == "fact":
                            lines.append(f"Факт: {content}")
                        elif role == "summary":
                            lines.append(f"Резюме: {content}")
                        else:
                            lines.append(content)
                    memory_context.append("Из памяти ИИ:\n" + "\n".join(lines))
            except Exception as e:
                logger.warning(f"Dialogue memory query failed: {e}")

        # Merge metadata with RAG and memory context
        batch_metadata = dict(metadata) if metadata else {}
        if rag_context:
            batch_metadata["rag_context"] = rag_context
        if memory_context:
            batch_metadata["memory_context"] = memory_context

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
            metadata=batch_metadata if batch_metadata else None,
        )

        # Store user message (check if we should skip storing to history)
        skip_history = metadata and metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )
            # Do NOT save raw messages to ChromaDB — AI decides via extraction

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        try:
            # agent.chat yields Union[SentenceOutput, Dict[str, Any]]
            agent_output_stream = context.agent_engine.chat(batch_input)

            async for output_item in agent_output_stream:
                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    # Handle SentenceOutput or AudioOutput
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await safe_websocket_send(
                websocket_send,
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                ),
            )
            # full_response will contain partial response before error
        # --- End processing agent response ---

        # Wait for any pending TTS tasks
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list)
            await safe_websocket_send(
                websocket_send, json.dumps({"type": "backend-synth-complete"})
            )

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        if context.history_uid and full_response:  # Check full_response before storing
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            # Do NOT save raw AI response to ChromaDB — AI extracts facts/summaries
            logger.info(f"AI response: {full_response}")

            # Schedule async background memory processing (extract facts, summarize)
            # Runs only when idle, does not block responses
            if context.dialogue_memory and context.history_uid:
                run_bg = getattr(context.agent_engine, "run_background_prompt", None)
                if callable(run_bg):

                    async def _bg_memory_task() -> None:
                        try:
                            msg_count = len(
                                get_history(
                                    context.character_config.conf_uid,
                                    context.history_uid,
                                )
                            )
                            await process_memory_background(
                                conf_uid=context.character_config.conf_uid,
                                history_uid=context.history_uid,
                                dialogue_memory=context.dialogue_memory,
                                llm_prompt_fn=run_bg,
                                message_count=msg_count,
                            )
                        except Exception as e:
                            logger.warning(f"Background memory task failed: {e}")

                    asyncio.create_task(_bg_memory_task())

        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await safe_websocket_send(
            websocket_send,
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"}),
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
