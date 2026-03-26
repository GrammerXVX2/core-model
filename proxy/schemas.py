from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    num_predict: Optional[int] = Field(default=None, description="Ollama-style max output tokens.")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(description="Message role.", examples=["user"])
    content: Any = Field(description="Message text/content payload.", examples=["Кратко объясни, что такое RAG."])


class ChatRequestModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = Field(default=None, description="Model name requested by client.")
    messages: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Conversation messages.",
        examples=[[{"role": "user", "content": "Сделай краткое резюме текста"}]],
    )
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=None, description="Requested max output tokens.")
    stream: bool = Field(default=False, description="When true, /api/chat returns an Ollama-compatible NDJSON stream.")
    reasoning: Optional[bool] = Field(
        default=None,
        description="Optional reasoning toggle for llama routes (q4/q6). Passed as enable_thinking.",
    )
    options: Optional[ChatOptions] = Field(default=None, description="Ollama options.")


class GenerateRequestModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = Field(default=None, description="Model name requested by client.")
    prompt: Optional[str] = Field(default=None, description="Input prompt.", examples=["Объясни разницу между REST и gRPC."])
    temperature: float = Field(default=0.7, description="Sampling temperature.")
    max_tokens: Optional[int] = Field(default=None, description="Requested max output tokens.")
    stream: bool = Field(default=False, description="Streaming is currently not supported on /api/generate.")
    options: Optional[ChatOptions] = Field(default=None, description="Ollama options.")


class EmbedRequestModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = Field(default=None, description="Optional embedding model alias (for example 4B or 8B route).")
    input: Optional[Any] = Field(
        default=None,
        description="Input text or list of texts for embedding.",
        examples=["Ошибка 502 при оплате картой"],
    )


class OllamaTextResponseModel(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    done_reason: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


class EmbedResponseModel(BaseModel):
    model: str
    embedding: List[float]
    embeddings: List[List[float]]
    total_duration: int
    load_duration: int
    prompt_eval_count: int


class ModelStatusItem(BaseModel):
    id: int = 0
    model: str
    model_vllm: str
    type: str
    base_url: str
    max_context_tokens: int
    status: str
    detail: str = ""


class ModelRegistryUpsertRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    public_model: str = Field(description="Public model alias exposed by proxy.")
    vllm_model: str = Field(description="Upstream backend model id.")
    model_type: Literal["chat", "embeddings", "reranker"] = Field(description="Endpoint type for this model.")
    base_url: str = Field(description="Upstream base URL, for example http://10.77.163.200:8000/v1")
    max_context_tokens: int = Field(ge=1, description="Maximum context window for this model.")
    default_max_tokens: int = Field(ge=1, description="Default max output tokens.")
    max_tokens_cap: int = Field(ge=1, description="Static output cap when dynamic mode is disabled.")
    min_context_headroom: int = Field(default=256, ge=0, description="Reserved context headroom.")
    stream_supported: bool = Field(default=False, description="Whether stream mode is supported.")
    reasoning_supported: bool = Field(default=False, description="Whether reasoning toggle is supported.")
    aliases: Optional[List[str]] = Field(default=None, description="Optional additional aliases.")


class ModelRegistryUpsertResponse(BaseModel):
    status: str
    model: ModelStatusItem
    aliases: List[str]


class ModelRegistryCrudPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    public_model: str
    vllm_model: str
    model_type: Literal["chat", "embeddings", "reranker"]
    base_url: str
    max_context_tokens: int = Field(ge=1)
    default_max_tokens: int = Field(ge=1)
    max_tokens_cap: int = Field(ge=1)
    min_context_headroom: int = Field(default=256, ge=0)
    stream_supported: bool = False
    reasoning_supported: bool = False
    aliases: Optional[List[str]] = None


class ModelRegistryItem(BaseModel):
    id: int
    public_model: str
    vllm_model: str
    model_type: Literal["chat", "embeddings", "reranker"]
    base_url: str
    max_context_tokens: int
    default_max_tokens: int
    max_tokens_cap: int
    min_context_headroom: int
    stream_supported: bool
    reasoning_supported: bool
    aliases: List[str]
    is_enabled: bool


class ModelRegistryCrudResponse(BaseModel):
    status: str
    model: ModelRegistryItem
