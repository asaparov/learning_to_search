from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from tokenizers import processors
from tokenizers.pre_tokenizers import BertPreTokenizer


def create_custom_tokenizer(vocab, save_path="./"):
    # Create a WordLevel tokenizer
    tokenizer = Tokenizer(WordLevel(vocab=dict(zip(vocab, range(len(vocab)))), unk_token="[UNK]"))
    
    # Set the pre-tokenizer to split on whitespace
    # tokenizer.pre_tokenizer = processors.Sequence([Whitespace(), Punctuation()])
    tokenizer.pre_tokenizer = BertPreTokenizer() #processors.Sequence([Whitespace(), Punctuation()])


    # Set up post-processing
    # tokenizer.post_processor = TemplateProcessing(
    #     single="[CLS] $A [SEP]",
    #     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #     special_tokens=[
    #         ("[CLS]", tokenizer.token_to_id("[CLS]")),
    #         ("[SEP]", tokenizer.token_to_id("[SEP]")),
    #     ],
    # )

    # Save the tokenizer
    tokenizer.save(f"{save_path}/tokenizer.json")
    print(f"Custom Tokenizer saved successfully to {save_path}/tokenizer.json")

    # Create a PreTrainedTokenizerFast from the tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        padding_side="left"
    )   

    # Save the tokenizer configuration
    fast_tokenizer.save_pretrained(save_path)
    
    return fast_tokenizer

def test_tokenizer(tokenizer):
    # Test encoding
    test_sentence = "Alex is a qumpus . Bob has shumpus ."
    encoded = tokenizer.encode(test_sentence)
    print(f"Encoded: {encoded}")

    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test tokenization
    tokens = tokenizer.tokenize(test_sentence)
    print(f"Tokens: {tokens}")

    # Test unknown words
    unknown_sentence = "Xavier likes supercalifragilisticexpialidocious zumpus ."
    unknown_encoded = tokenizer.encode(unknown_sentence)
    print(f"Encoded with unknown word: {unknown_encoded}")
    unknown_decoded = tokenizer.decode(unknown_encoded)
    print(f"Decoded with unknown word: {unknown_decoded}")

# Create the custom tokenizer

from vocab import NAMES, NOUNS, CONNECTORS, VOCAB
from test_tk import create_custom_tokenizer
tokenizer = create_custom_tokenizer(VOCAB)


if __name__ == "__main__":
    from vocab import VOCAB
    tokenizer = create_custom_tokenizer(VOCAB)

    # # Test the tokenizer
    # print("\nTesting the tokenizer:")
    # test_tokenizer(tokenizer)

    # # Load the tokenizer from saved files and test again
    # print("\nTesting the loaded tokenizer:")
    # loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained("./")
    # test_tokenizer(loaded_tokenizer)