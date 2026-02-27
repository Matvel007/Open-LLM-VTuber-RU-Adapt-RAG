# config_manager/system.py
from pydantic import Field, model_validator
from typing import Dict, ClassVar

from .i18n import I18nMixin, Description
from .rag import RAGConfig


class SystemConfig(I18nMixin):
    """System configuration settings."""

    conf_version: str = Field(..., alias="conf_version")
    host: str = Field(..., alias="host")
    port: int = Field(..., alias="port")
    config_alts_dir: str = Field(..., alias="config_alts_dir")
    tool_prompts: Dict[str, str] = Field(..., alias="tool_prompts")
    enable_proxy: bool = Field(False, alias="enable_proxy")
    auto_start_microphone: bool = Field(True, alias="auto_start_microphone")
    launch_pet_mode_only: bool = Field(False, alias="launch_pet_mode_only")
    rag_config: RAGConfig | None = Field(default=None, alias="rag_config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "conf_version": Description(en="Configuration version", zh="配置文件版本"),
        "host": Description(en="Server host address", zh="服务器主机地址"),
        "port": Description(en="Server port number", zh="服务器端口号"),
        "config_alts_dir": Description(
            en="Directory for alternative configurations", zh="备用配置目录"
        ),
        "tool_prompts": Description(
            en="Tool prompts to be inserted into persona prompt",
            zh="要插入到角色提示词中的工具提示词",
        ),
        "enable_proxy": Description(
            en="Enable proxy mode for multiple clients",
            zh="启用代理模式以支持多个客户端使用一个 ws 连接",
        ),
        "auto_start_microphone": Description(
            en="Auto-start microphone when client connects (server sends start-mic)",
            zh="客户端连接时自动启动麦克风（服务端发送 start-mic）",
        ),
        "launch_pet_mode_only": Description(
            en="Launch only in pet/desktop overlay mode (no main window). Client must support this.",
            zh="仅以宠物/桌面覆盖模式启动（无主窗口）。客户端需支持此选项。",
        ),
        "rag_config": Description(
            en="RAG (Retrieval-Augmented Generation) settings with ChromaDB",
            zh="RAG（检索增强生成）设置，使用 ChromaDB",
        ),
    }

    @model_validator(mode="after")
    def check_port(cls, values):
        port = values.port
        if port < 0 or port > 65535:
            raise ValueError("Port must be between 0 and 65535")
        return values
