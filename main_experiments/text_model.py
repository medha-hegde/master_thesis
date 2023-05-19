# Text Model Set up
def load_text_model(config):

    pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
    print("Loading Model %s" % pretrained_model_name_or_path)

    if 'clip' in pretrained_model_name_or_path:
        from transformers import CLIPTextModel, CLIPTokenizer

        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path)

    elif 'byt5' in pretrained_model_name_or_path:
        from transformers import T5EncoderModel, ByT5Tokenizer

        text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path)
        tokenizer = ByT5Tokenizer.from_pretrained(pretrained_model_name_or_path)

    elif 't5' in pretrained_model_name_or_path:
        from transformers import T5Tokenizer, T5EncoderModel

        text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)

    text_encoder.requires_grad_(False)
    text_encoder.to('cuda')

    return tokenizer, text_encoder
