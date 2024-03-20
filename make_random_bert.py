from transformers import BertConfig, BertForMaskedLM

if __name__=="__main__":
	#config = BertConfig(
	#    vocab_size=50000,
	#    hidden_size=768,
	#    num_hidden_layers=6, 
	#    num_attention_heads=12,
	#    max_position_embeddings=512
	#)
	config = BertConfig.from_pretrained('avichr/heBERT')
    #config=BertConfig.from_pretrained('avichr/heBERT')
	model = BertForMaskedLM(config)
	model.save_pretrained('random_hebert')
