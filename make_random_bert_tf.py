from transformers import BertConfig, TFBertForMaskedLM

if __name__=="__main__":
    config = BertConfig(
        vocab_size=50000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        max_position_embeddings=512
    )

    model = TFBertForMaskedLM(config)
    model.build()
    model.save_pretrained('tf_random_bert')                                                   
