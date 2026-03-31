MODEL_STATUS_AVAILABLE = "доступен"
MODEL_STATUS_UNAVAILABLE = "недоступен"
MODEL_STATUS_WARMING = "прогрев"
MODEL_STATUS_POLLING_DETAIL = "model status poll in progress"

ERR_NO_CHAT_MODELS = "no chat models registered in status cache"
ERR_NO_EMBEDDING_MODELS = "no embedding models registered in status cache"
ERR_NO_MODELS_REGISTERED = "no models registered in model registry"
ERR_MODEL_UNAVAILABLE_FMT = "model unavailable: {model} at {base_url}. detail: {detail}"
ERR_MODEL_DOES_NOT_SUPPORT_ENDPOINT_FMT = "model does not support {endpoint} endpoint: {model}"
ERR_UNSUPPORTED_MODEL_FOR_ENDPOINT_FMT = "unsupported model for {endpoint}; allowed: {allowed}"
ERR_MODEL_DOES_NOT_SUPPORT_VISION_FMT = "model does not support vision input: {model}"
ERR_MODEL_DOES_NOT_SUPPORT_REASONING_FMT = "model does not support reasoning toggle: {model}"
ERR_MODEL_DOES_NOT_SUPPORT_STREAM_FMT = "model does not support stream mode: {model}"

OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"
OPENAI_EMBEDDINGS_PATH = "/embeddings"
OPENAI_V1_EMBEDDINGS_PATH = "/v1/embeddings"
TEI_EMBED_PATH = "/embed"
TEI_V1_EMBED_PATH = "/v1/embed"

EMBEDDING_PATH_CANDIDATES = (
    OPENAI_EMBEDDINGS_PATH,
    TEI_EMBED_PATH,
    OPENAI_V1_EMBEDDINGS_PATH,
    TEI_V1_EMBED_PATH,
)

LOG_CHAT_INCOMING = "chat.incoming body=%s"
LOG_CHAT_ADAPTED = "chat.adapted model=%s route_model=%s base_url=%s messages=%s roles=%s est_input_tokens=%s max_tokens=%s"
LOG_CHAT_VLLM_RESPONSE = "chat.vllm_response finish_reason=%s content_preview=%s reasoning_preview=%s"
LOG_CHAT_UI_ADAPTED = "chat.ui.adapted model=%s route_model=%s base_url=%s messages=%s est_input_tokens=%s max_tokens=%s"

LOG_EMBED_INCOMING = "embed.incoming body=%s"
LOG_EMBED_ADAPTED = "embed.adapted model=%s route_model=%s base_url=%s input_type=%s input_preview=%s"
LOG_EMBED_VLLM_ERROR = "embed.vllm_error status=%s detail=%s"
LOG_EMBED_VLLM_RESPONSE = "embed.vllm_response vectors=%s first_dim=%s prompt_tokens=%s"

LOG_MODELS_POLLER_ERROR = "models.poller.error=%s"

LOG_REQ_PARSE_START = "req.parse.start path=%s content_type=%s content_length=%s"
LOG_REQ_PARSE_JSON_ERROR = "req.parse.json_error path=%s error=%s"
LOG_REQ_PARSE_FORM_ERROR = "req.parse.form_urlencoded_error path=%s error=%s"
LOG_REQ_PARSE_MULTIPART_ERROR = "req.parse.multipart_error path=%s error=%s"
LOG_REQ_PARSE_RAW_PREVIEW = "req.parse.raw_preview path=%s raw=%s"
LOG_REQ_PARSE_KEYS = "req.parse.keys path=%s keys=%s"
LOG_REQ_PARSE_EMPTY_DICT = "req.parse.empty_dict path=%s"
OPENAI_COMPLETIONS_PATH = "/completions"
