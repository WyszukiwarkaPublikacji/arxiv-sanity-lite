images_index_type = "IVF_FLAT"
image_embedding_size = 512
image_metric_type = "IP"
caption_embedding_size = 384
caption_metric_type = "IP"

embedding_batch_size = 32
extraction_batch_size = 64
rendering_min_caption_length = 15
rendering_dpi = 120

chemical_index_type = "BIN_FLAT"
chemical_embedding_size = 2048

# in production should probably be Eventually, but Strong might make testing easier
consistency_level = "Strong"
