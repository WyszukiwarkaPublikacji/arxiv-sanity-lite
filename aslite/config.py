images_index_type = "IVF_FLAT"
chart_embedding_size = 1024
chart_metric_type = "IP"
caption_embedding_size = 384
caption_metric_type = "IP"

model_input_size = 416
embedding_batch_size = 32
extraction_batch_size = 64
rendering_ensure_captions = True
rendering_dpi = 120

chemical_index_type = "BIN_FLAT"
chemical_embedding_size: int = 2048

# in production should probably be Eventually, but Strong might make testing easier
consistency_level = "Strong"
