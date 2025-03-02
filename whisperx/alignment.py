"""
Forced Alignment with Whisper
C. Max Bain
"""
import math

from dataclasses import dataclass
from typing import Iterable, Optional, Union, List

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .audio import SAMPLE_RATE, load_audio
from .utils import interpolate_nans
from .types import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SegmentData,
)
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

PUNKT_ABBREVIATIONS = ['dr', 'vs', 'mr', 'mrs', 'prof']

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
}


def load_align_model(language_code: str, device: str, model_name: Optional[str] = None, model_dir=None):
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir)
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata


def align(
        transcript: Iterable[SingleSegment],
        model: torch.nn.Module,
        align_model_metadata: dict,
        audio: Union[str, np.ndarray, torch.Tensor],
        device: str,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
        combined_progress: bool = False,
        logger=None,
        batch_size: int = 8,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription using batched inference.
    """

    # Helper function to log messages
    def log_message(message, level="info"):
        if logger:
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
            elif level == "debug":
                logger.debug(message)
        if print_progress:
            print(message)

    # Load audio if needed
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # Get the blank token ID
    blank_id = 0
    for char, code in model_dictionary.items():
        if char == '[pad]' or char == '<pad>':
            blank_id = code

    # Convert transcript to list if it's not already
    transcript_list = list(transcript)
    total_segments = len(transcript_list)

    # Initialize return structure with default values
    aligned_segments = []

    # ====== PHASE 1: Preprocess all segments ======
    log_message("Phase 1: Preprocessing segments")

    # Create a list of segments to process
    segments_to_process = []

    for sdx, segment in enumerate(transcript_list):
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100 * 0.5  # 50% for preprocessing
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            log_message(f"Preprocessing: {percent_complete:.2f}%...")

        try:
            # Validate segment
            if "text" not in segment or not isinstance(segment["text"], str) or not segment["text"].strip():
                log_message(f"Skipping segment {sdx}: missing or empty text", level="warning")
                aligned_segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", ""),
                    "words": [],
                    "chars": None if not return_char_alignments else []
                })
                continue

            if "start" not in segment or "end" not in segment:
                log_message(f"Skipping segment {sdx}: missing start/end times", level="warning")
                aligned_segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", ""),
                    "words": [],
                    "chars": None if not return_char_alignments else []
                })
                continue

            t1, t2 = segment["start"], segment["end"]

            # Check timing
            if t1 >= MAX_DURATION:
                log_message(f"Skipping segment {sdx}: start time {t1} exceeds audio duration {MAX_DURATION}",
                            level="warning")
                aligned_segments.append({
                    "start": t1,
                    "end": t2,
                    "text": segment["text"],
                    "words": [],
                    "chars": None if not return_char_alignments else []
                })
                continue

            # Preprocess text
            text = segment["text"]
            num_leading = len(text) - len(text.lstrip())
            num_trailing = len(text) - len(text.rstrip())

            # Clean characters
            clean_char, clean_cdx = [], []
            for cdx, char in enumerate(text):
                char_ = char.lower()
                if model_lang not in LANGUAGES_WITHOUT_SPACES:
                    char_ = char_.replace(" ", "|")

                if cdx < num_leading or cdx > len(text) - num_trailing - 1:
                    pass
                elif char_ in model_dictionary.keys():
                    clean_char.append(char_)
                    clean_cdx.append(cdx)
                else:
                    clean_char.append('*')
                    clean_cdx.append(cdx)

            # Skip if no alignable content
            if not any(c in model_dictionary for c in clean_char):
                log_message(f"Skipping segment {sdx}: no alignable characters", level="warning")
                aligned_segments.append({
                    "start": t1,
                    "end": t2,
                    "text": text,
                    "words": [],
                    "chars": None if not return_char_alignments else []
                })
                continue

            # Get word indices
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                per_word = text.split(" ")
            else:
                per_word = text

            clean_wdx = []
            for wdx, wrd in enumerate(per_word):
                if any([c in model_dictionary.keys() for c in wrd.lower()]):
                    clean_wdx.append(wdx)
                else:
                    clean_wdx.append(wdx)

            # Extract sentence spans
            try:
                punkt_param = PunktParameters()
                punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
                sentence_splitter = PunktSentenceTokenizer(punkt_param)
                sentence_spans = list(sentence_splitter.span_tokenize(text))
                if not sentence_spans:
                    sentence_spans = [(0, len(text))]
            except Exception as e:
                log_message(f"Error in sentence splitting for segment {sdx}: {e}", level="warning")
                sentence_spans = [(0, len(text))]

            # Extract audio segment
            f1 = int(t1 * SAMPLE_RATE)
            f2 = min(int(t2 * SAMPLE_RATE), audio.shape[1])
            waveform_segment = audio[:, f1:f2]

            # Handle minimum length requirement
            if waveform_segment.shape[1] < 400:
                waveform_segment = torch.nn.functional.pad(waveform_segment, (0, 400 - waveform_segment.shape[1]))

            # Convert text to tokens
            text_clean = "".join(clean_char)
            tokens = [model_dictionary.get(c, -1) for c in text_clean]

            # Store processed segment data
            segments_to_process.append({
                "sdx": sdx,
                "segment": segment,
                "waveform": waveform_segment,
                "text": text,
                "text_clean": text_clean,
                "tokens": tokens,
                "clean_char": clean_char,
                "clean_cdx": clean_cdx,
                "clean_wdx": clean_wdx,
                "sentence_spans": sentence_spans,
                "t1": t1,
                "t2": t2
            })

            # Create placeholder for result
            aligned_segments.append(None)

        except Exception as e:
            log_message(f"Error preprocessing segment {sdx}: {e}", level="warning")
            aligned_segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment.get("text", ""),
                "words": [],
                "chars": None if not return_char_alignments else []
            })

    # ====== PHASE 2: Process batches ======
    log_message(f"Phase 2: Processing {len(segments_to_process)} valid segments in batches of {batch_size}")

    # Create batches (simple sequential batches)
    num_batches = (len(segments_to_process) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(segments_to_process))
        batch = segments_to_process[start_idx:end_idx]

        if print_progress:
            base_progress = ((batch_idx + 1) / num_batches) * 100 * 0.5  # 50% for batch processing
            percent_complete = (50 + 50 + base_progress / 2) if combined_progress else (50 + base_progress)
            log_message(f"Processing batch {batch_idx + 1}/{num_batches}: {percent_complete:.2f}%...")

        try:
            # Stack waveforms for batch processing
            # First, find the longest waveform in the batch to determine padding size
            max_length = max(seg["waveform"].shape[1] for seg in batch)

            # Pad all waveforms to the same length
            padded_waveforms = []
            for seg in batch:
                waveform = seg["waveform"]
                if waveform.shape[1] < max_length:
                    padding = (0, max_length - waveform.shape[1])
                    waveform = torch.nn.functional.pad(waveform, padding)
                padded_waveforms.append(waveform)

            # Stack waveforms
            batch_waveforms = torch.cat(padded_waveforms, dim=0)

            # Run model inference
            with torch.inference_mode():
                if model_type == "torchaudio":
                    emissions_batch, _ = model(batch_waveforms.to(device))
                elif model_type == "huggingface":
                    emissions_batch = model(batch_waveforms.to(device)).logits
                else:
                    raise NotImplementedError(f"Align model of type {model_type} not supported.")

                emissions_batch = torch.log_softmax(emissions_batch, dim=-1)

            # Process each segment's result
            for i, segment_data in enumerate(batch):
                emission = emissions_batch[i].cpu().detach()

                # Process this segment
                try:
                    process_result = process_single_segment(
                        segment_data,
                        emission,
                        blank_id,
                        model_dictionary,
                        model_lang,
                        return_char_alignments,
                        interpolate_method
                    )

                    aligned_segments[segment_data["sdx"]] = process_result
                except Exception as e:
                    log_message(f"Error processing segment {segment_data['sdx']}: {e}", level="warning")
                    aligned_segments[segment_data["sdx"]] = {
                        "start": segment_data["t1"],
                        "end": segment_data["t2"],
                        "text": segment_data["text"],
                        "words": [],
                        "chars": None if not return_char_alignments else []
                    }

        except Exception as e:
            log_message(f"Error processing batch {batch_idx}: {e}", level="warning")
            # Fall back to individual processing
            for segment_data in batch:
                try:
                    # Process one by one
                    with torch.inference_mode():
                        # Use single segment waveform
                        waveform = segment_data["waveform"].to(device)

                        if model_type == "torchaudio":
                            emission, _ = model(waveform)
                        elif model_type == "huggingface":
                            emission = model(waveform).logits
                        else:
                            raise NotImplementedError(f"Align model of type {model_type} not supported.")

                        emission = torch.log_softmax(emission, dim=-1)
                        emission = emission[0].cpu().detach()

                        process_result = process_single_segment(
                            segment_data,
                            emission,
                            blank_id,
                            model_dictionary,
                            model_lang,
                            return_char_alignments,
                            interpolate_method
                        )

                        aligned_segments[segment_data["sdx"]] = process_result
                except Exception as e:
                    log_message(f"Error in fallback processing for segment {segment_data['sdx']}: {e}", level="warning")
                    aligned_segments[segment_data["sdx"]] = {
                        "start": segment_data["t1"],
                        "end": segment_data["t2"],
                        "text": segment_data["text"],
                        "words": [],
                        "chars": None if not return_char_alignments else []
                    }

    # Fill in any None values
    for i in range(len(aligned_segments)):
        if aligned_segments[i] is None:
            segment = transcript_list[i]
            aligned_segments[i] = {
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment.get("text", ""),
                "words": [],
                "chars": None if not return_char_alignments else []
            }

    # Create word_segments list
    word_segments = []
    for segment in aligned_segments:
        if "words" in segment:
            word_segments.extend(segment["words"])

    log_message(f"Alignment completed. Processed {len(aligned_segments)} segments with {len(word_segments)} words.")
    return {"segments": aligned_segments, "word_segments": word_segments}


def process_single_segment(
        segment_data,
        emission,
        blank_id,
        model_dictionary,
        model_lang,
        return_char_alignments,
        interpolate_method
):
    """
    Process a single segment's alignment after model inference.

    Args:
        segment_data: Dictionary with preprocessed segment data
        emission: Model emission probabilities
        blank_id: ID of the blank token
        model_dictionary: Dictionary mapping characters to indices
        model_lang: Language code
        return_char_alignments: Whether to return character-level alignments
        interpolate_method: Method to use for interpolating missing timestamps

    Returns:
        Dictionary with aligned segment data
    """
    segment = segment_data["segment"]
    text = segment_data["text"]
    tokens = segment_data["tokens"]
    text_clean = segment_data["text_clean"]
    t1 = segment_data["t1"]
    t2 = segment_data["t2"]
    clean_cdx = segment_data["clean_cdx"]
    clean_wdx = segment_data["clean_wdx"]
    sentence_spans = segment_data["sentence_spans"]

    # Initialize result structure
    aligned_seg = {
        "start": t1,
        "end": t2,
        "text": text,
        "words": [],
        "chars": None if not return_char_alignments else []
    }

    # Get trellis matrix
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)

    if path is None:
        return aligned_seg

    # Merge repeats
    char_segments = merge_repeats(path, text_clean)

    # Calculate alignment ratio
    duration = t2 - t1
    ratio = duration * emission.size(0) / (trellis.size(0) - 1)

    # Map character segments to original text
    char_segments_arr = []
    word_idx = 0
    for cdx, char in enumerate(text):
        start, end, score = None, None, None
        if cdx in clean_cdx:
            char_seg = char_segments[clean_cdx.index(cdx)]
            start = round(char_seg.start * ratio + t1, 3)
            end = round(char_seg.end * ratio + t1, 3)
            score = round(char_seg.score, 3)

        char_segments_arr.append({
            "char": char,
            "start": start,
            "end": end,
            "score": score,
            "word-idx": word_idx
        })

        # Update word index
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            word_idx += 1
        elif cdx == len(text) - 1 or text[cdx + 1] == " ":
            word_idx += 1

    # Convert to DataFrame for easier processing
    char_segments_arr = pd.DataFrame(char_segments_arr)

    # Process sentences
    aligned_subsegments = []

    # Assign sentence index to characters
    char_segments_arr["sentence-idx"] = None
    for sdx, (sstart, send) in enumerate(sentence_spans):
        # Get characters in this sentence span
        curr_chars = char_segments_arr.loc[
            (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]

        if curr_chars.empty:
            continue

        char_segments_arr.loc[
            (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx

        sentence_text = text[sstart:send]

        # Handle empty characters
        if curr_chars["start"].dropna().empty or curr_chars["end"].dropna().empty:
            aligned_subsegments.append({
                "text": sentence_text,
                "start": t1,
                "end": t2,
                "words": [],
                "chars": [] if return_char_alignments else None
            })
            continue

        sentence_start = curr_chars["start"].min()
        end_chars = curr_chars[curr_chars["char"] != ' ']
        sentence_end = end_chars["end"].max() if not end_chars.empty else t2

        # Process words in this sentence
        sentence_words = []
        for word_idx in sorted(curr_chars["word-idx"].unique()):
            word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
            word_text = "".join(word_chars["char"].tolist()).strip()

            if not word_text:
                continue

            # Skip spaces for alignment
            word_chars = word_chars[word_chars["char"] != " "]

            if word_chars.empty or word_chars["start"].dropna().empty or word_chars["end"].dropna().empty:
                continue

            word_start = word_chars["start"].min()
            word_end = word_chars["end"].max()
            word_score = round(word_chars["score"].mean(), 3)

            word_segment = {"word": word_text}

            if not np.isnan(word_start):
                word_segment["start"] = word_start
            if not np.isnan(word_end):
                word_segment["end"] = word_end
            if not np.isnan(word_score):
                word_segment["score"] = word_score

            sentence_words.append(word_segment)

        # Store sentence data
        sentence_data = {
            "text": sentence_text,
            "start": sentence_start,
            "end": sentence_end,
            "words": sentence_words,
        }

        # Add character alignments if requested
        if return_char_alignments:
            curr_chars = curr_chars[["char", "start", "end", "score"]]
            curr_chars.fillna(-1, inplace=True)
            curr_chars = curr_chars.to_dict("records")
            curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
            sentence_data["chars"] = curr_chars

        aligned_subsegments.append(sentence_data)

    # Handle empty aligned subsegments
    if not aligned_subsegments:
        return aligned_seg

    # Convert to DataFrame for grouping
    try:
        aligned_subsegments = pd.DataFrame(aligned_subsegments)

        # Interpolate missing timestamps
        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)

        # Group sentences with same timestamps
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"

        aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
        aligned_subsegments = aligned_subsegments.to_dict('records')

        return aligned_subsegments[0] if len(aligned_subsegments) == 1 else aligned_seg
    except Exception as e:
        # Fall back to the original segment
        return aligned_seg

"""
source: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
"""


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1:, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            # trellis[t, :-1] + emission[t, tokens[1:]],
            trellis[t, :-1] + get_wildcard_emission(emission[t], tokens[1:], blank_id),
        )
    return trellis


def get_wildcard_emission(frame_emission, tokens, blank_id):
    """Processing token emission scores containing wildcards (vectorized version)

    Args:
        frame_emission: Emission probability vector for the current frame
        tokens: List of token indices
        blank_id: ID of the blank token

    Returns:
        tensor: Maximum probability score for each token position
    """
    assert 0 <= blank_id < len(frame_emission)

    # Convert tokens to a tensor if they are not already
    tokens = torch.tensor(tokens) if not isinstance(tokens, torch.Tensor) else tokens

    # Create a mask to identify wildcard positions
    wildcard_mask = (tokens == -1)

    # Get scores for non-wildcard positions
    regular_scores = frame_emission[tokens.clamp(min=0)]  # clamp to avoid -1 index

    # Create a mask and compute the maximum value without modifying frame_emission
    max_valid_score = frame_emission.clone()   # Create a copy
    max_valid_score[blank_id] = float('-inf')  # Modify the copy to exclude the blank token
    max_valid_score = max_valid_score.max()

    # Use where operation to combine results
    result = torch.where(wildcard_mask, max_valid_score, regular_scores)

    return result


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        # p_change = emission[t - 1, tokens[j]]
        p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]



@dataclass
class Path:
    points: List[Point]
    score: float


@dataclass
class BeamState:
    """State in beam search."""
    token_index: int   # Current token position
    time_index: int    # Current time step
    score: float       # Cumulative score
    path: List[Point]  # Path history


def backtrack_beam(trellis, emission, tokens, blank_id=0, beam_width=5):
    """Standard CTC beam search backtracking implementation.

    Args:
        trellis (torch.Tensor): The trellis (or lattice) of shape (T, N), where T is the number of time steps
                                and N is the number of tokens (including the blank token).
        emission (torch.Tensor): The emission probabilities of shape (T, N).
        tokens (List[int]): List of token indices (excluding the blank token).
        blank_id (int, optional): The ID of the blank token. Defaults to 0.
        beam_width (int, optional): The number of top paths to keep during beam search. Defaults to 5.

    Returns:
        List[Point]: the best path
    """
    T, J = trellis.size(0) - 1, trellis.size(1) - 1

    init_state = BeamState(
        token_index=J,
        time_index=T,
        score=trellis[T, J],
        path=[Point(J, T, emission[T, blank_id].exp().item())]
    )

    beams = [init_state]

    while beams and beams[0].token_index > 0:
        next_beams = []

        for beam in beams:
            t, j = beam.time_index, beam.token_index

            if t <= 0:
                continue

            p_stay = emission[t - 1, blank_id]
            p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]

            stay_score = trellis[t - 1, j]
            change_score = trellis[t - 1, j - 1] if j > 0 else float('-inf')

            # Stay
            if not math.isinf(stay_score):
                new_path = beam.path.copy()
                new_path.append(Point(j, t - 1, p_stay.exp().item()))
                next_beams.append(BeamState(
                    token_index=j,
                    time_index=t - 1,
                    score=stay_score,
                    path=new_path
                ))

            # Change
            if j > 0 and not math.isinf(change_score):
                new_path = beam.path.copy()
                new_path.append(Point(j - 1, t - 1, p_change.exp().item()))
                next_beams.append(BeamState(
                    token_index=j - 1,
                    time_index=t - 1,
                    score=change_score,
                    path=new_path
                ))

        # sort by score
        beams = sorted(next_beams, key=lambda x: x.score, reverse=True)[:beam_width]

        if not beams:
            break

    if not beams:
        return None

    best_beam = beams[0]
    t = best_beam.time_index
    j = best_beam.token_index
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        best_beam.path.append(Point(j, t - 1, prob))
        t -= 1

    return best_beam.path[::-1]


# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words
