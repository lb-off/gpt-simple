"""
Tokenizer wrapper for gpt_simple.

Wraps HuggingFace tokenizers (GPT-2, SentencePiece, or any AutoTokenizer) with
consistent special-token handling (EOS, PAD, EOD) and metaspace hotfixes.
"""

import logging
import os
import json
from typing import List, Optional, Union

import torch
from transformers import GPT2Tokenizer, GPT2TokenizerFast, AutoTokenizer

logger = logging.getLogger("gpt_simple")


class SimpleLLMTokenizer:
    """Thin wrapper around a HuggingFace tokenizer.

    Handles:
    - Auto-detection of tokenizer format (BPE, SentencePiece, etc.)
    - Consistent special tokens (EOS, PAD, EOD)
    - SentencePiece empty-metaspace hotfix
    """

    def __init__(self, model_name: str = "gpt2"):
        if not model_name:
            logger.warning("No tokenizer path provided, defaulting to 'gpt2'")
            model_name = "gpt2"

        logger.debug(f"Loading tokenizer from: {model_name}")

        try:
            if os.path.exists(model_name):
                self.tokenizer = self._load_local(model_name)
            else:
                self.tokenizer = self._load_hub(model_name)
        except Exception as e:
            logger.warning(f"Error loading tokenizer from '{model_name}': {e}. Falling back to 'gpt2'")
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load both custom and fallback tokenizers. "
                    f"Custom error: {e}, Fallback error: {e2}"
                )

        self._setup_pad_token()
        self._setup_token_ids()
        self._detect_metaspace()
        self._log_summary()

    # -- loading helpers ----------------------------------------------------

    @staticmethod
    def _load_local(model_name: str):
        """Detect format in a local directory and load the right class."""
        files_in_dir = os.listdir(model_name) if os.path.isdir(model_name) else []
        logger.debug(f"Local tokenizer path: {model_name} (files: {files_in_dir})")

        if any(f.endswith(".model") for f in files_in_dir) or any(
            "tokenizer.model" in f for f in files_in_dir
        ):
            return SimpleLLMTokenizer._load_sentencepiece(model_name, files_in_dir)

        if "tokenizer.json" in files_in_dir:
            logger.debug("Detected Byte-Level BPE format (tokenizer.json)")
            tokenizer_json_path = os.path.join(model_name, "tokenizer.json")
            return GPT2TokenizerFast(tokenizer_file=tokenizer_json_path)

        if "vocab.json" in files_in_dir and "merges.txt" in files_in_dir:
            logger.debug("Detected GPT-2 format (vocab.json + merges.txt)")
            return GPT2Tokenizer.from_pretrained(model_name)

        logger.debug("Unknown tokenizer format, trying AutoTokenizer")
        return AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def _load_sentencepiece(model_name: str, files_in_dir: list):
        """Load a SentencePiece-based tokenizer."""
        logger.debug("Detected SentencePiece tokenizer format")

        config_path = os.path.join(model_name, "config.json")
        if not os.path.exists(config_path):
            logger.debug("config.json missing, creating minimal one for SentencePiece loading")
            try:
                config_data = {"model_type": "llama", "tokenizer_class": "LlamaTokenizer"}
                with open(config_path, "w") as f:
                    json.dump(config_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not create config.json: {e}")

        try:
            tok = AutoTokenizer.from_pretrained(model_name, legacy=False)
            return tok
        except Exception as e1:
            logger.debug(f"AutoTokenizer failed for SentencePiece: {e1}")

        for cls_name in ("LlamaTokenizer", "T5Tokenizer", "AlbertTokenizer", "XLNetTokenizer"):
            try:
                logger.debug(f"Trying {cls_name}...")
                import transformers
                cls = getattr(transformers, cls_name)
                tok = cls.from_pretrained(model_name)
                logger.debug(f"Loaded with {cls_name}")
                return tok
            except Exception:
                continue

        raise RuntimeError("Could not load SentencePiece tokenizer with any available method")

    @staticmethod
    def _load_hub(model_name: str):
        logger.debug(f"Loading from HuggingFace Hub: {model_name}")
        try:
            return GPT2Tokenizer.from_pretrained(model_name)
        except Exception:
            logger.debug("GPT2Tokenizer failed, trying AutoTokenizer")
            return AutoTokenizer.from_pretrained(model_name)

    # -- special-token setup ------------------------------------------------

    def _setup_pad_token(self):
        try:
            needs_pad = not hasattr(self.tokenizer, "pad_token") or self.tokenizer.pad_token is None
            if needs_pad:
                try:
                    added = self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                    if added and added > 0:
                        logger.debug(f"Added pad token '<|pad|>' ({added} token(s) added)")
                except Exception as e_add:
                    if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        logger.debug(f"Using eos_token as pad_token fallback ({e_add})")
                    else:
                        logger.warning(f"Could not add PAD token and no eos_token available: {e_add}")
        except Exception as e:
            logger.warning(f"Could not configure pad_token: {e}")

    def _setup_token_ids(self):
        self.vocab_size = len(self.tokenizer)
        try:
            self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
            self.bos_token_id = getattr(self.tokenizer, "bos_token_id", None) or self.eos_token_id
            self.eod_token_id = self.eos_token_id
        except Exception as e:
            logger.warning(f"Could not get special token IDs: {e}")
            self.eos_token_id = None
            self.pad_token_id = None
            self.bos_token_id = None
            self.eod_token_id = None

    def _detect_metaspace(self):
        """Find the empty metaspace token in SentencePiece tokenizers."""
        self.empty_metaspace_token_id = None
        is_byte_level_bpe = isinstance(self.tokenizer, GPT2TokenizerFast)

        if not is_byte_level_bpe and hasattr(self.tokenizer, "convert_ids_to_tokens"):
            for token_id in range(min(1000, self.vocab_size), min(self.vocab_size, 50000)):
                try:
                    token_str = self.tokenizer.convert_ids_to_tokens([token_id])
                    if isinstance(token_str, list) and len(token_str) > 0:
                        token_str = token_str[0]
                    if token_str == "\u2581":
                        decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
                        if decoded == "" or decoded.strip() == "":
                            self.empty_metaspace_token_id = token_id
                            logger.debug(f"Found empty metaspace token: ID {token_id}")
                            break
                except Exception:
                    continue

    def _log_summary(self):
        parts = [f"{type(self.tokenizer).__name__} (vocab={self.vocab_size}"]
        if self.pad_token_id is not None:
            parts.append(f"pad={self.pad_token_id}")
        if self.eos_token_id is not None:
            parts.append(f"eos={self.eos_token_id}")
        if self.eod_token_id is not None and self.eod_token_id != self.eos_token_id:
            parts.append(f"eod={self.eod_token_id}")
        logger.debug("Loaded " + ", ".join(parts) + ")")

    # -- public API ---------------------------------------------------------

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], List[List[int]], torch.Tensor]:
        """Encode text to token IDs."""
        if isinstance(text, str):
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding="max_length" if padding and max_length else False,
                truncation=truncation,
                return_tensors=return_tensors,
            )
            if isinstance(tokens, list) and self.empty_metaspace_token_id is not None:
                filtered = []
                for t in tokens:
                    if t == self.empty_metaspace_token_id:
                        if 71 < self.vocab_size:
                            filtered.append(71)
                    else:
                        filtered.append(t)
                tokens = filtered
        else:
            tokens = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding="max_length" if padding else False,
                truncation=truncation,
                return_tensors=return_tensors,
            )
            if hasattr(tokens, "input_ids"):
                tokens = tokens.input_ids
        return tokens

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> Union[str, List[str]]:
        """Decode token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids is None:
            return ""
        if isinstance(token_ids, list) and len(token_ids) == 0:
            return ""

        is_llama = (
            hasattr(self.tokenizer, "__class__")
            and "Llama" in self.tokenizer.__class__.__name__
            and not isinstance(self.tokenizer, GPT2TokenizerFast)
        )

        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            results: List[str] = []
            for ids in token_ids:
                if ids is None or (isinstance(ids, list) and len(ids) == 0):
                    results.append("")
                else:
                    decoded = self.tokenizer.decode(
                        ids,
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                    if is_llama and decoded and not decoded.isspace():
                        decoded = decoded.rstrip(" ")
                    results.append(decoded)
            return results
        else:
            decoded = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            if is_llama and decoded and not decoded.isspace():
                decoded = decoded.rstrip(" ")
            return decoded

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into sub-word strings (not IDs)."""
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)
