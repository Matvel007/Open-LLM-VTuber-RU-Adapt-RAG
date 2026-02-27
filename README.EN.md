![](./assets/banner.jpg)

<h1 align="center">Open-LLM-VTuber — RU-Adapt-RAG</h1>
<h3 align="center">

[![Original Project](https://img.shields.io/badge/Original-Open--LLM--VTuber-blue?style=flat&logo=github)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber)
[![license](https://img.shields.io/badge/license-MIT-green?style=flat)](https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/blob/master/LICENSE)

[Русский](README.md) | **English**

</h3>

## ⭐️ About

This is a **mod** of **Open-LLM-VTuber** with Russian support, RAG system, and extra features. Voice AI companion with Live2D avatar, speech recognition, and visual perception. Runs offline. Platform: Windows.

### 👀 Demo

![](assets/demo_screenshot.png)

**Videos:**

![Video 1](https://github.com/Matvel007/Open-LLM-VTuber-RU-Adapt-RAG/raw/main/assets/demo_video_1.mp4)

![Video 2](https://github.com/Matvel007/Open-LLM-VTuber-RU-Adapt-RAG/raw/main/assets/demo_video_2.mp4)

## ✨ What's new in RU-Adapt-RAG mod

- **Auto-start in pet mode** — app launches directly in pet mode without the main window
- **Microphone off at startup** — enable manually
- **RAG system window** — dedicated window for RAG setup: connect documents and chats, semantic search
- **RAG menus** — quick access to settings and knowledge base management
- **Easy Live2D model installation** — add and switch characters via menu
- **Light Live2D loading** — models load in the background, interface stays responsive
- **BGE-M3 embedding model (BorisTM/bge-m3_en_ru)** — multilingual model, suitable for Russian and English
- **No emotions in text** — removed emotions that the original version wrote in responses (e.g. *smiles*), now only natural speech

## 📚 RAG system

RAG is the character's "memory". It remembers your dialogues and what you show it in documents. When you ask a question, the character searches for similar topics in chats and files and answers using that context. You get a living interlocutor who remembers conversations and draws on your materials instead of replying from scratch. All RAG management is done directly from settings: you can add documents, connect chats, and configure the knowledge base without editing any files.

## 🔊 TTS (speech)

**Silero TTS** — quality local Russian voice, fully offline. Models v5_1_ru, v5_ru, v4_ru load via torch.hub.

## 🎭 Quick Live2D model switching

The new menu lets you switch characters without editing configs. Put the Live2D model folder in `live2d-models/` — it will show up in the menu, and you can switch in a couple of clicks. Easy to try different characters and pick the right one without extra steps.

## 📌 Pinning the character on screen

In pet mode you can pin the character to a spot on screen and choose the plane: **front** (above all windows) or **back** (under windows, visible only on the desktop background). Drag the character where you like and choose whether it should always be visible or stay in the background.

## 🚀 Run

### Build
See [Quick Start](https://open-llm-vtuber.github.io/docs/quick-start). Install deps: `uv sync`. Build frontend if needed in `frontend/`.

### Run via bat
```bat
start.bat
```
Or `start_ru.bat` for Russian config.

### Run exe (desktop client)
After build — exe in `frontend/release/`. Run the built exe; it will connect to the server.

### Run server manually
```bat
uv run run_server.py
```

## 🤗 Contribute

[Development guide](https://docs.llmvtuber.com/docs/development-guide/overview)

## 📜 Licenses

Live2D models are licensed separately by Live2D Inc.
