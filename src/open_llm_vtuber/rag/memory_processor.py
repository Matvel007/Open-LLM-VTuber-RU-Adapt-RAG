"""Async background memory processing: extract facts, summarize, update profile."""

import re
from typing import Callable, Awaitable

from loguru import logger

from ..chat_history_manager import get_history
from .dialogue_memory import DialogueMemory

MEMORY_LOG_PREFIX = "[Память]"

# Prompts for LLM-based extraction (Russian)
# Факты — важное, что нужно помнить и не забыть (имя, предпочтения, интересы)
FACT_EXTRACTION_SYSTEM = """Ты — помощник по извлечению важных фактов из диалога.
Сохраняй ТОЛЬКО то, что нужно помнить о пользователе и что важно не забыть.

Сохраняй: имя, возраст, город, профессия, увлечения, предпочтения, важные события.
НЕ сохраняй: приветствия, пустые фразы, темы разговоров (это идёт в резюме).

Верни список фактов, по одному на строку, без нумерации.
Формат: "Имя: X", "Любит: Y" и т.п. Если нет содержательных фактов — верни пустую строку."""

# Резюме — о чём говорили в диалогах, контекст разговоров
SUMMARY_SYSTEM = """Ты — помощник по суммаризации диалога.
Сделай краткое резюме (2-4 предложения): о чём говорили, какие темы обсуждали.
Это запись о прошедших диалогах. Выдели главное. Пропускай приветствия."""

# Профиль — 1 комплексная запись: память AI о пользователе
PROFILE_UPDATE_SYSTEM = """Ты — помощник по составлению профиля пользователя.
На основе списка фактов сформируй единый комплексный профиль — как AI помнит пользователя:
имя, город, профессия, интересы, предпочтения и т.д.
Верни краткий текст (5-10 пунктов) для контекста AI."""

PROFILE_MERGE_SYSTEM = """Ты — помощник по слиянию фактов в единый профиль пользователя.
Профиль — это 1 комплексная запись: вся память AI о пользователе.
Факты даны от новых к старым (сверху — новее).

ВАЖНО: При противоречиях используй НОВЫЙ факт. Пример: "Любит торты" → "Не любит торты" → итог: "Не любит торты".
Сформируй единый профиль без противоречий. Кратко (5-15 пунктов).
Формат: по одному пункту на строку, например "Имя: X", "Любит: Y"."""

FACT_EXTRACT_EVERY = 3
SUMMARY_EVERY = 6


async def process_memory_background(
    conf_uid: str,
    history_uid: str,
    dialogue_memory: DialogueMemory | None,
    llm_prompt_fn: Callable[[list[dict[str, str]], str], Awaitable[str]],
    message_count: int,
) -> None:
    """
    Run memory processing in background: extract facts, summarize, update profile.

    Called after each conversation turn completes. Does not block the main flow.
    Uses llm_prompt_fn(messages, system) -> full_response to call the LLM.

    Args:
        conf_uid: Character config ID.
        history_uid: Chat history ID.
        dialogue_memory: DialogueMemory instance (or None if disabled).
        llm_prompt_fn: Async function (messages, system) -> full text response.
        message_count: Total messages in history (used to decide when to summarize).
    """
    if not dialogue_memory or not history_uid or not conf_uid:
        return

    try:
        messages_raw = get_history(conf_uid, history_uid)
        if not messages_raw:
            return

        # Build messages for LLM (last N exchanges)
        last_n = min(10, len(messages_raw))
        recent = messages_raw[-last_n:]
        dialog_text = "\n".join(
            f"{'Пользователь' if m['role'] == 'human' else 'Ассистент'}: {m.get('content', '')}"
            for m in recent
        )
        if not dialog_text.strip():
            return

        # Extract facts every N messages (3, 6, 9, ...)
        if (
            message_count >= FACT_EXTRACT_EVERY
            and (message_count - FACT_EXTRACT_EVERY) % FACT_EXTRACT_EVERY == 0
        ):
            try:
                facts_text = await llm_prompt_fn(
                    [{"role": "user", "content": f"Диалог:\n{dialog_text}"}],
                    FACT_EXTRACTION_SYSTEM,
                )
                if facts_text and facts_text.strip():
                    facts = [
                        f.strip()
                        for f in re.split(r"[\n•\-]", facts_text)
                        if f.strip() and len(f.strip()) > 3
                    ]
                    for fact in facts[:10]:  # Limit to 10 facts per batch
                        dialogue_memory.add_item(
                            role="fact",
                            content=fact,
                            history_uid=history_uid,
                            conf_uid=conf_uid,
                        )
                    # Merge all facts into profile, newer overrides older
                    all_facts = dialogue_memory.get_all_facts(history_uid, conf_uid)
                    if all_facts:
                        merge_prompt = "Факты (сверху — новее):\n" + "\n".join(
                            f"- {f}" for f in all_facts[:30]
                        )
                        try:
                            merged_profile = await llm_prompt_fn(
                                [{"role": "user", "content": merge_prompt}],
                                PROFILE_MERGE_SYSTEM,
                            )
                            if merged_profile and merged_profile.strip():
                                dialogue_memory.set_user_profile(
                                    history_uid, conf_uid, merged_profile
                                )
                                logger.info(
                                    f"{MEMORY_LOG_PREFIX} Профиль обновлён (противоречия сняты)."
                                )
                        except Exception as e:
                            logger.warning(f"Profile merge failed: {e}")
                    logger.info(
                        f"{MEMORY_LOG_PREFIX} Извлечено {len(facts)} фактов из диалога."
                    )
            except Exception as e:
                logger.warning(f"Fact extraction failed: {e}")

        # Summarize every N messages (6, 12, 18, ...)
        if (
            message_count >= SUMMARY_EVERY
            and (message_count - SUMMARY_EVERY) % SUMMARY_EVERY == 0
        ):
            try:
                summary_text = await llm_prompt_fn(
                    [{"role": "user", "content": f"Диалог:\n{dialog_text}"}],
                    SUMMARY_SYSTEM,
                )
                if summary_text and summary_text.strip():
                    dialogue_memory.add_item(
                        role="summary",
                        content=summary_text.strip(),
                        history_uid=history_uid,
                        conf_uid=conf_uid,
                    )
                    logger.info(
                        f'{MEMORY_LOG_PREFIX} Сохранено summary: "{summary_text[:50]}..."'
                    )
            except Exception as e:
                logger.warning(f"Summary extraction failed: {e}")

    except Exception as e:
        logger.warning(f"Background memory processing failed: {e}")
