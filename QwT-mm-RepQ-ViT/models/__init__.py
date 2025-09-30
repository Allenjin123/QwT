from . import clip

def load_model(args):
    # load model
    model, preprocess = clip.load(args.model, device=args.device)
    # load tokenizer
    tokenizer = clip.tokenize
    return model, preprocess, tokenizer