import polars as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from src.constants import AllWords, MODEL_FILES_PATH
from typing import Optional

from transformers import (
    TrOCRConfig,
    TrOCRForCausalLM,
    ViTConfig,
    ViTModel,
    VisionEncoderDecoderModel,
    DeiTImageProcessor,
    AutoTokenizer,
)


class TrOcrDataset(Dataset):
    def __init__(self, df, image_processes, tokenizer, test_mode=False):
        self.df = df
        self.image_processes = image_processes
        self.tokenizer = tokenizer
        self.test_mode = test_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.df["path"][idx])
        except UnidentifiedImageError:
            image = Image.new("RGB", (224, 224))

        pixel_values = self.image_processes(image, return_tensors="pt").pixel_values
        if self.test_mode:
            return {"pixel_values": pixel_values.squeeze()}

        target = self.df["target"][idx]
        labels = self.tokenizer(target, padding="max_length", max_length=17).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.tokenizer.pad_token_id else -100 for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
            "target": target,
        }
        return encoding


class TrOcrModel:
    def __init__(
        self,
        pretraining_model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = ViTModel(ViTConfig())
        decoder_config = TrOCRConfig()
        decoder_config.vocab_size = 1000
        decoder_config.d_model = 256
        decoder_config.decoder_attention_heads = 8
        decoder_config.decoder_ffn_dim = 1024
        decoder_config.decoder_layers = 6
        decoder = TrOCRForCausalLM(decoder_config)

        self.model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

        if len(tokenizer_name):
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_FILES_PATH / tokenizer_name
            )
        else:
            self.tokenizer = self.create_tokenizer()

        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.config.eos_token_id = self.tokenizer.sep_token_id

        self.model.config.max_length = 17
        self.model.config.early_stopping = True
        self.model.config.num_beams = 5
        self.model.config.do_sample = False
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 1

        if len(pretraining_model_name):
            self.model.load_state_dict(
                torch.load(
                    MODEL_FILES_PATH / pretraining_model_name, map_location="cpu"
                )
            )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def create_tokenizer():
        def get_training_corpus(words):
            for start_idx in range(0, len(words), 1000):
                yield str(words[start_idx: start_idx + 1000])

        old_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        new_tokenizer = old_tokenizer.train_new_from_iterator(
            get_training_corpus(AllWords.word_list), 1000, min_frequence=1
        )
        new_tokenizer.save_pretrained(MODEL_FILES_PATH / "tokenizer")
        return new_tokenizer

    def save_model(self, model_name: str) -> None:
        self.model.save(self.model.state_dict(), MODEL_FILES_PATH / model_name)

    def predict(
        self, test_data: pl.DataFrame, test_mode: bool = True, batch_size: int = 16
    ) -> pl.DataFrame:
        test_dataset = TrOcrDataset(
            test_data, DeiTImageProcessor({}), self.tokenizer, test_mode
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        results, scores = [], []
        for x in tqdm(test_dataloader, total=len(test_dataloader)):
            generated_ids = self.model.generate(
                x["pixel_values"].to(self.device),
                num_beams=15,
                length_penalty=0,
                output_scores=True,
                return_dict_in_generate=True,
                num_return_sequences=15,
                do_sample=False,
                early_stopping=False,
            )
            sequences_scores = (
                generated_ids["sequences_scores"].reshape(-1, 15).tolist()
            )
            generated_ids = generated_ids["sequences"].reshape(
                -1, 15, generated_ids["sequences"].shape[1]
            )
            set_all_words = set(AllWords.word_list)
            for scrs, ids in zip(sequences_scores, generated_ids):
                generated_text = self.tokenizer.batch_decode(
                    ids, skip_special_tokens=True
                )
                generated_text = [x.replace(" ", "") for x in generated_text]
                ind = [i for i, x in enumerate(generated_text) if x in set_all_words]
                results.append([generated_text[i] for i in ind])
                scores.append([scrs[i] for i in ind])
        test_data = test_data.with_columns(pl.Series(name="predict", values=results))
        test_data = test_data.with_columns(pl.Series(name="score", values=scores))
        return test_data
