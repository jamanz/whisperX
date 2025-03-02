"""
Optimized Forced Alignment with Wav2Vec2
"""
import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Assuming these imports are from the original module
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

# Constants from the original code
PUNKT_ABBREVIATIONS = ['dr', 'vs', 'mr', 'mrs', 'prof']
LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

# Update the default French model to our preferred one
DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "jonatasgrosman/wav2vec2-large-fr-voxpopuli-french",  # Changed from "VOXPOPULI_ASR_BASE_10K_FR"
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

# Keep the original dictionaries for other languages
DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    # Other languages remain unchanged...
}

# Create a directory for storing quantized models
QUANTIZED_MODELS_DIR = os.environ.get("QUANTIZED_MODELS_DIR", "./quantized_models")
os.makedirs(QUANTIZED_MODELS_DIR, exist_ok=True)

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

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

@dataclass
class BeamState:
    """State in beam search."""
    token_index: int
    time_index: int
    score: float
    path: List[Point]


def load_align_model(
    language_code: str,
    device: str,
    model_name: Optional[str] = None,
    model_dir=None,
    use_quantized: bool = True
):
    """
    Load and optimize the alignment model for the specified language.

    Args:
        language_code: Language code for the model
        device: Device to load the model on ('cpu' or 'cuda')
        model_name: Optional specific model name to use
        model_dir: Directory for model cache
        use_quantized: Whether to use quantized model for CPU inference

    Returns:
        Tuple of (model, metadata)
    """
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).")
            raise ValueError(f"No default align-model for language: {language_code}")

    # Create a safe filename for the quantized model
    safe_model_name = model_name.replace("/", "_").replace(":", "_")
    quantized_path = os.path.join(QUANTIZED_MODELS_DIR, f"{safe_model_name}_quantized.pt")

    # For torchaudio pipeline models
    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]

        if use_quantized and device == "cpu" and os.path.exists(quantized_path):
            try:
                print(f"Loading pre-quantized model from {quantized_path}")
                align_model = torch.load(quantized_path)
            except Exception as e:
                print(f"Error loading quantized model: {e}, falling back to non-quantized")
                align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir})
        else:
            align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir})
            if use_quantized and device == "cpu":
                try:
                    print("Quantizing model (one-time process)...")
                    start_time = time.time()
                    align_model = torch.quantization.quantize_dynamic(
                        align_model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    torch.save(align_model, quantized_path)
                    print(f"Model quantized in {time.time() - start_time:.2f} seconds")
                except Exception as eq:
                    print(f"Quantization failed: {eq}, using original model")

        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}

    # For huggingface models
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir)

            if use_quantized and device == "cpu" and os.path.exists(quantized_path):
                try:
                    print(f"Loading pre-quantized model from {quantized_path}")
                    align_model = torch.load(quantized_path)
                except Exception as e:
                    print(f"Error loading quantized model: {e}, falling back to non-quantized")
                    align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir)
            else:
                align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir)
                if use_quantized and device == "cpu":
                    try:
                        print("Quantizing model (one-time process)...")
                        start_time = time.time()
                        # Move to CPU for quantization if needed
                        if align_model.device.type == "cuda":
                            align_model = align_model.cpu()

                        align_model = torch.quantization.quantize_dynamic(
                            align_model, {torch.nn.Linear}, dtype=torch.qint8
                        )
                        torch.save(align_model, quantized_path)
                        print(f"Model quantized in {time.time() - start_time:.2f} seconds")
                    except Exception as eq:
                        print(f"Quantization failed: {eq}, using original model")

        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found')

        pipeline_type = "huggingface"
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char, code in processor.tokenizer.get_vocab().items()}

    # Move to the desired device
    align_model = align_model.to(device)

    # Try to optimize with JIT for non-quantized models on CUDA
    if device == "cuda" and not use_quantized:
        try:
            # Try to optimize with JIT compilation for CUDA
            align_model = torch.jit.script(align_model)
            print("Successfully applied JIT optimization to the model")
        except Exception as e:
            print(f"JIT optimization failed: {e}, using standard model")

    align_metadata = {
        "language": language_code,
        "dictionary": align_dictionary,
        "type": pipeline_type
    }

    return align_model, align_metadata


def process_batch(model, waveform_batch, lengths_batch, model_type, device):
    """Process a batch of audio segments at once"""
    with torch.inference_mode():
        try:
            if model_type == "torchaudio":
                emissions_batch, _ = model(waveform_batch.to(device), lengths=lengths_batch)
            elif model_type == "huggingface":
                emissions_batch = model(waveform_batch.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            return torch.log_softmax(emissions_batch, dim=-1)
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Fall back to processing one by one if batch fails
            results = []
            for i in range(waveform_batch.shape[0]):
                waveform = waveform_batch[i:i+1]
                length = None if lengths_batch is None else lengths_batch[i:i+1]
                if model_type == "torchaudio":
                    emission, _ = model(waveform.to(device), lengths=length)
                else:
                    emission = model(waveform.to(device)).logits
                results.append(torch.log_softmax(emission, dim=-1))
            return torch.cat(results, dim=0)


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
            trellis[t, :-1] + get_wildcard_emission(emission[t], tokens[1:], blank_id),
        )
    return trellis


def get_wildcard_emission(frame_emission, tokens, blank_id):
    """Processing token emission scores containing wildcards (vectorized version)"""
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


def backtrack_beam(trellis, emission, tokens, blank_id=0, beam_width=5):
    """Optimized beam search backtracking implementation."""
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

        # Sort by score and keep only top beams
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


def safe_min(values):
    """Get min value safely handling NaN or empty values"""
    if len(values) == 0:
        return None
    try:
        min_val = values.min()
        return None if pd.isna(min_val) else min_val
    except:
        return None


def safe_max(values):
    """Get max value safely handling NaN or empty values"""
    if len(values) == 0:
        return None
    try:
        max_val = values.max()
        return None if pd.isna(max_val) else max_val
    except:
        return None

def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    batch_size: int = 8,  # Adjust based on GPU memory
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    num_workers: int = 4,  # For parallel preprocessing
    logger = None,
    beam_width: int = 2,  # Beam search width
    max_retry: int = 2,    # Number of retries for model inference
) -> AlignedTranscriptionResult:
    """
    Optimized version of align function that uses batching and parallel processing

    Args:
        transcript: List of transcript segments
        model: The alignment model
        align_model_metadata: Model metadata dictionary
        audio: Audio input as path, numpy array, or tensor
        device: Device to use for inference ('cpu' or 'cuda')
        batch_size: Number of segments to process at once
        interpolate_method: Method for interpolating timestamps
        return_char_alignments: Whether to return character-level alignments
        print_progress: Whether to print progress updates
        combined_progress: Flag for combined progress reporting
        num_workers: Number of workers for parallel preprocessing
        logger: Optional logger instance
        beam_width: Width for beam search

    Returns:
        AlignedTranscriptionResult: Dictionary with aligned segments and word segments
    """
    # Helper function for logging
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

    # Prepare audio input with better error handling
    try:
        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                try:
                    audio = load_audio(audio)
                except Exception as e:
                    log_message(f"Error loading audio file: {str(e)}", level="error")
                    return {"segments": [], "word_segments": []}
            audio = torch.from_numpy(audio)
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
    except Exception as e:
        log_message(f"Error processing audio input: {str(e)}", level="error")
        return {"segments": [], "word_segments": []}

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]
    total_segments = len(transcript)
    start_time = time.time()
    # Handle empty transcript
    transcript_list = list(transcript)
    if len(transcript_list) == 0:
        log_message("Empty transcript provided", level="warning")
        return {"segments": [], "word_segments": []}

    def preprocess_segment(sdx):
        try:
            segment = transcript_list[sdx]
            # strip spaces at beginning / end, but keep track of the amount.
            num_leading = len(segment["text"]) - len(segment["text"].lstrip())
            num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
            text = segment["text"]

            # Handle empty text or extremely short text
            if len(text.strip()) == 0:
                return sdx, {
                    "can_align": False,
                    "reason": "empty_text"
                }

            # Check for excessively long segments (warning only)
            if segment["end"] - segment["start"] > 30.0:  # 30 seconds
                log_message(f"Segment {sdx} is very long ({segment['end'] - segment['start']:.1f}s), " 
                            f"alignment may be less accurate", level="warning")

            # split into words
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                per_word = text.split(" ")
            else:
                per_word = text

            clean_char, clean_cdx = [], []
            for cdx, char in enumerate(text):
                try:
                    char_ = char.lower()
                    # wav2vec2 models use "|" character to represent spaces
                    if model_lang not in LANGUAGES_WITHOUT_SPACES:
                        char_ = char_.replace(" ", "|")

                    # ignore whitespace at beginning and end of transcript
                    if cdx < num_leading:
                        pass
                    elif cdx > len(text) - num_trailing - 1:
                        pass
                    elif char_ in model_dictionary.keys():
                        clean_char.append(char_)
                        clean_cdx.append(cdx)
                    else:
                        # add placeholder
                        clean_char.append('*')
                        clean_cdx.append(cdx)
                except Exception as char_e:
                    # Handle individual character errors gracefully
                    log_message(f"Error processing character at position {cdx}: {char_e}", level="debug")
                    clean_char.append('*')
                    clean_cdx.append(cdx)

            clean_wdx = []
            for wdx, wrd in enumerate(per_word):
                if any([c in model_dictionary.keys() for c in wrd.lower()]):
                    clean_wdx.append(wdx)
                else:
                    # index for placeholder
                    clean_wdx.append(wdx)

            try:
                punkt_param = PunktParameters()
                punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
                sentence_splitter = PunktSentenceTokenizer(punkt_param)
                sentence_spans = list(sentence_splitter.span_tokenize(text))
            except Exception as e:
                # Fallback if sentence splitting fails
                log_message(f"Sentence splitting failed for segment {sdx}, using whole segment as one sentence", level="debug")
                sentence_spans = [(0, len(text))]

            text_clean = "".join(clean_char)
            tokens = [model_dictionary.get(c, -1) for c in text_clean]

            return sdx, {
                "clean_char": clean_char,
                "clean_cdx": clean_cdx,
                "clean_wdx": clean_wdx,
                "sentence_spans": sentence_spans,
                "text_clean": text_clean,
                "can_align": len(clean_char) > 0 and segment["start"] < MAX_DURATION,
                "tokens": tokens
            }
        except Exception as e:
            log_message(f"Error in preprocessing segment {sdx}: {e}", level="warning")
            return sdx, {"can_align": False, "reason": str(e)}

    # Use ThreadPoolExecutor for preprocessing
    segment_data = {}

    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(preprocess_segment, i) for i in range(total_segments)]
            for future in as_completed(futures):
                idx, data = future.result()
                segment_data[idx] = data
                if print_progress and len(segment_data) % max(1, total_segments // 10) == 0:
                    log_message(f"Preprocessing progress: {len(segment_data) / total_segments * 100:.1f}%", level="debug")
    else:
        for i in range(total_segments):
            idx, data = preprocess_segment(i)
            segment_data[idx] = data
            if print_progress and (i+1) % max(1, total_segments // 10) == 0:
                log_message(f"Preprocessing progress: {(i+1) / total_segments * 100:.1f}%", level="debug")

    # 2. Group segments for batch processing
    batches = []
    batch_indices = []
    batch_lengths = []
    segment_groups = {}  # Group segments by similar length to improve batching efficiency

    # First, group segments by similar length (rounded to nearest second)
    for idx in range(total_segments):
        if not segment_data[idx]["can_align"]:
            continue

        segment = transcript_list[idx]
        duration = int(round((segment["end"] - segment["start"]) * SAMPLE_RATE / 1000))  # Length in frames, rounded
        if duration not in segment_groups:
            segment_groups[duration] = []
        segment_groups[duration].append(idx)

    # Process each length group
    for _, indices in segment_groups.items():
        current_batch = []
        current_indices = []
        current_lengths = []

        for idx in indices:
            segment = transcript_list[idx]
            t1 = segment["start"]
            t2 = segment["end"]
            f1 = int(t1 * SAMPLE_RATE)
            f2 = int(t2 * SAMPLE_RATE)

            waveform_segment = audio[:, f1:f2]
            # Handle minimum length requirement
            if waveform_segment.shape[-1] < 400:
                waveform_segment = torch.nn.functional.pad(
                    waveform_segment, (0, 400 - waveform_segment.shape[-1])
                )

            current_batch.append(waveform_segment)
            current_indices.append(idx)
            current_lengths.append(waveform_segment.shape[-1])

            if len(current_batch) >= batch_size:
                # Pad all segments to the maximum length in this batch
                max_length = max(current_lengths)
                padded_batch = []
                for i, segment in enumerate(current_batch):
                    if segment.shape[-1] < max_length:
                        segment = torch.nn.functional.pad(
                            segment, (0, max_length - segment.shape[-1])
                        )
                    padded_batch.append(segment)

                # Process this batch
                batches.append(torch.cat(padded_batch, dim=0))
                batch_indices.append(current_indices)
                batch_lengths.append(current_lengths)
                current_batch = []
                current_indices = []
                current_lengths = []

        # Add remaining segments in this length group
        if current_batch:
            # Pad all segments to the maximum length in this batch
            max_length = max(current_lengths)
            padded_batch = []
            for i, segment in enumerate(current_batch):
                if segment.shape[-1] < max_length:
                    segment = torch.nn.functional.pad(
                        segment, (0, max_length - segment.shape[-1])
                    )
                padded_batch.append(segment)

            batches.append(torch.cat(padded_batch, dim=0))
            batch_indices.append(current_indices)
            batch_lengths.append(current_lengths)

    # 3. Process batches
    aligned_segments = [None] * total_segments

    for batch_idx, (batch_audio, indices) in enumerate(zip(batches, batch_indices)):
        if print_progress:
            base_progress = ((batch_idx + 1) / len(batches)) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            log_message(f"Processing batch {batch_idx+1}/{len(batches)} - Progress: {percent_complete:.2f}%...", level="info")

        # Get model predictions for the batch
        batch_start_time = time.time()
        with torch.inference_mode():
            try:
                emissions_batch = process_batch(model, batch_audio, None, model_type, device)
            except Exception as e:
                log_message(f"Failed to process batch: {str(e)}", level="error")
                continue

        # Process each segment in the batch
        for batch_pos, segment_idx in enumerate(indices):
            segment = transcript_list[segment_idx]
            emission = emissions_batch[batch_pos].cpu().detach()

            # Create the aligned segment structure
            aligned_seg = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "words": [],
                "chars": None if not return_char_alignments else [],
            }

            # Find blank_id
            blank_id = 0
            for char, code in model_dictionary.items():
                if char == '[pad]' or char == '<pad>':
                    blank_id = code

            # Get trellis and path with better error handling
            try:
                tokens = segment_data[segment_idx]["tokens"]
                trellis = get_trellis(emission, tokens, blank_id)
            except Exception as e:
                log_message(f'Error in trellis calculation for segment "{segment["text"]}": {str(e)}', level="error")
                aligned_segments[segment_idx] = aligned_seg
                continue

            try:
                path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=beam_width)
            except Exception as e:
                log_message(f'Error in backtracking for segment "{segment["text"]}": {str(e)}', level="error")
                aligned_segments[segment_idx] = aligned_seg
                continue

            if path is None:
                log_message(f'Failed to align segment "{segment["text"]}": backtrack failed', level="warning")
                aligned_segments[segment_idx] = aligned_seg
                continue

            # Process the aligned path
            text_clean = segment_data[segment_idx]["text_clean"]
            char_segments = merge_repeats(path, text_clean)

            # Calculate duration and ratio
            t1 = segment["start"]
            t2 = segment["end"]
            duration = t2 - t1
            ratio = duration * batch_audio.size(1) / (trellis.size(0) - 1)

            # Assign timestamps to aligned characters
            text = segment["text"]
            char_segments_arr = []
            word_idx = 0
            for cdx, char in enumerate(text):
                start, end, score = None, None, None
                if cdx in segment_data[segment_idx]["clean_cdx"]:
                    char_seg = char_segments[segment_data[segment_idx]["clean_cdx"].index(cdx)]
                    start = round(char_seg.start * ratio + t1, 3)
                    end = round(char_seg.end * ratio + t1, 3)
                    score = round(char_seg.score, 3)

                char_segments_arr.append(
                    {
                        "char": char,
                        "start": start,
                        "end": end,
                        "score": score,
                        "word-idx": word_idx,
                    }
                )

                # Increment word_idx
                if model_lang in LANGUAGES_WITHOUT_SPACES:
                    word_idx += 1
                elif cdx == len(text) - 1 or text[cdx+1] == " ":
                    word_idx += 1

            char_segments_arr = pd.DataFrame(char_segments_arr)

            aligned_subsegments = []
            try:
                # Assign sentence_idx to each character index
                char_segments_arr["sentence-idx"] = None
                for sdx2, (sstart, send) in enumerate(segment_data[segment_idx]["sentence_spans"]):
                    curr_chars = char_segments_arr.loc[
                        (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
                    char_segments_arr.loc[
                        (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx2

                    sentence_text = text[sstart:send]
                    sentence_start = curr_chars["start"].min()
                    end_chars = curr_chars[curr_chars["char"] != ' ']
                    sentence_end = end_chars["end"].max()
                    sentence_words = []

                    for word_idx in curr_chars["word-idx"].unique():
                        word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                        word_text = "".join(word_chars["char"].tolist()).strip()
                        if len(word_text) == 0:
                            continue

                        # dont use space character for alignment
                        word_chars = word_chars[word_chars["char"] != " "]

                        word_start = safe_min(word_chars["start"])
                        word_end = safe_max(word_chars["end"])

                        # Calculate score carefully to avoid NaN issues
                        valid_scores = word_chars["score"].dropna()
                        word_score = round(valid_scores.mean(), 3) if len(valid_scores) > 0 else None

                        # -1 indicates unalignable
                        word_segment = {"word": word_text}

                        if word_start is not None:
                            word_segment["start"] = word_start
                        if word_end is not None:
                            word_segment["end"] = word_end
                        if word_score is not None:
                            word_segment["score"] = word_score

                        sentence_words.append(word_segment)

                    aligned_subsegments.append({
                        "text": sentence_text,
                        "start": sentence_start,
                        "end": sentence_end,
                        "words": sentence_words,
                    })

                    if return_char_alignments:
                        curr_chars = curr_chars[["char", "start", "end", "score"]]
                        curr_chars.fillna(-1, inplace=True)
                        curr_chars = curr_chars.to_dict("records")
                        curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
                        aligned_subsegments[-1]["chars"] = curr_chars

                aligned_subsegments = pd.DataFrame(aligned_subsegments)
                # Make sure we have valid start/end timestamps
                if len(aligned_subsegments) > 0:
                    # Handle interpolation carefully
                    try:
                        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
                        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
                    except Exception as e:
                        log_message(f"Error interpolating timestamps: {e}. Using original segment timestamps.", level="warning")
                        # Fallback to segment timestamps if interpolation fails
                        aligned_subsegments["start"] = aligned_subsegments["start"].fillna(segment["start"])
                        aligned_subsegments["end"] = aligned_subsegments["end"].fillna(segment["end"])

                    # concatenate sentences with same timestamps
                    try:
                        agg_dict = {"text": " ".join, "words": "sum"}
                        if model_lang in LANGUAGES_WITHOUT_SPACES:
                            agg_dict["text"] = "".join
                        if return_char_alignments:
                            agg_dict["chars"] = "sum"
                        aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
                    except Exception as e:
                        log_message(f"Error grouping sentences: {e}. Using individual sentences.", level="warning")

                    aligned_subsegments = aligned_subsegments.to_dict('records')
                    aligned_segments[segment_idx] = aligned_subsegments
                else:
                    log_message(f"No valid subsegments for segment {segment_idx}", level="warning")
                    aligned_segments[segment_idx] = aligned_seg

            except Exception as e:
                log_message(f'Error processing aligned characters for segment "{segment["text"]}": {str(e)}',
                           level="error")
                aligned_segments[segment_idx] = aligned_seg

        # Free memory explicitly
        del emissions_batch
        if device == "cuda":
            torch.cuda.empty_cache()

    # 4. Prepare the final result
    # Fill in default alignments for non-alignable segments
    final_aligned_segments = []
    for idx, segment in enumerate(transcript_list):
        if aligned_segments[idx] is None:
            # Use default alignment
            aligned_seg = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "words": [],
                "chars": None if not return_char_alignments else [],
            }
            final_aligned_segments.append(aligned_seg)
        elif isinstance(aligned_segments[idx], list):
            # Multiple subsegments
            final_aligned_segments.extend(aligned_segments[idx])
        else:
            # Single segment
            final_aligned_segments.append(aligned_segments[idx])

    # Create word_segments list
    word_segments = []
    for segment in final_aligned_segments:
        if "words" in segment:
            word_segments.extend(segment["words"])

    total_time = time.time() - start_time
    log_message(f"Alignment completed in {total_time:.2f} seconds. Processed {len(final_aligned_segments)} segments with {len(word_segments)} words.",
               level="info")

    return {"segments": final_aligned_segments, "word_segments": word_segments}