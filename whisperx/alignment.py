import math
from dataclasses import dataclass
from typing import Iterable, Optional, Union, List, Dict

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.utils import interpolate_nans
from whisperx.types import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SegmentData,
)
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import numba as nb

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
    # Additional models...
}

def load_align_model(language_code: str, device: Union[str, List[str]], model_name: Optional[str] = None, model_dir=None):
    """
    Load alignment model with support for multiple devices.
    
    Args:
        language_code: Language code for model selection
        device: Either a single device string or list of device strings (e.g. ["cuda:0", "cuda:1"])
        model_name: Optional specific model name
        model_dir: Optional model directory
        
    Returns:
        If device is a string (single GPU): (model, metadata)
        If device is a list (multi-GPU): (models dict, metadata)
    """
    # Handle the case where we want to load models on multiple devices
    multi_device = isinstance(device, list)
    devices = device if multi_device else [device]
    
    # Select the model based on language code
    if model_name is None:
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\
                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")
    
    # Dictionary to hold models for each device
    models = {}
    
    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        
        # Load model on each device
        for dev in devices:
            models[dev] = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(dev)
            
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir)
            
            # Load model on each device
            for dev in devices:
                models[dev] = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir).to(dev)
                
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        
        pipeline_type = "huggingface"
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char, code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    # Return model(s) and metadata
    if multi_device:
        return models, align_metadata
    else:
        return models[devices[0]], align_metadata

# wrong end timestampts + long processing
def align_batch(
        transcript: Iterable[SingleSegment],
        model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
        align_model_metadata: dict,
        audio: Union[str, np.ndarray, torch.Tensor],
        device: Union[str, List[str]],
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
        combined_progress: bool = False,
        logger=None,
        batch_size: int = 4,  # Default batch size of 4
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    Optimized version with batch processing, reduced GPU synchronization, and faster backtracking.
    Supports multi-GPU processing when device is a list of device strings.
    """
    import time
    start_time_total = time.time()

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

    # Check if we're using multiple devices
    multi_device = isinstance(device, list)
    devices = device if multi_device else [device]
    models = model if multi_device else {devices[0]: model}

    log_message(f"Using {'multiple devices' if multi_device else 'single device'}: {devices}")

    # Create CUDA streams for each device
    streams = {dev: torch.cuda.Stream(device=dev) for dev in devices}

    # Time loading and preparing audio
    start_time_audio = time.time()
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    audio_prep_time = time.time() - start_time_audio
    log_message(f"Audio preparation took {audio_prep_time:.4f} seconds")

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    start_time_preprocess = time.time()
    transcript_list = list(transcript)  # Convert to list if it's an iterator
    total_segments = len(transcript_list)

    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript_list):
        # Progress reporting
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            log_message(f"Preprocessing Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
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

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd.lower()]):
                clean_wdx.append(wdx)
            else:
                # index for placeholder
                clean_wdx.append(wdx)

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }

    preprocess_time = time.time() - start_time_preprocess
    log_message(f"Preprocessing took {preprocess_time:.4f} seconds for {total_segments} segments")

    aligned_segments: List[SingleAlignedSegment] = []

    # Time for the main alignment loop
    start_time_alignment = time.time()
    model_inference_time_total = 0
    trellis_time_total = 0
    backtrack_time_total = 0
    postprocess_time_total = 0
    batch_prep_time_total = 0
    batch_count = 0

    # Find blank_id once
    blank_id = 0
    for char, code in model_dictionary.items():
        if char == '[pad]' or char == '<pad>':
            blank_id = code

    # Group segments by device to reduce context switching
    device_segments = [[] for _ in range(len(devices))]
    valid_segments = []

    # Pre-filter segments to remove invalid ones
    for sdx, segment in enumerate(transcript_list):
        t1 = segment["start"]

        aligned_seg = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": [],
            "chars": None,
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # Early validation checks
        if len(segment_data[sdx]["clean_char"]) == 0:
            log_message(f'Failed to align segment ("{segment["text"]}"): '
                        f'no characters in this segment found in model dictionary, '
                        f'resorting to original...', level="warning")
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            log_message(f'Failed to align segment ("{segment["text"]}"): '
                        f'original start time longer than audio duration, skipping...', level="warning")
            aligned_segments.append(aligned_seg)
            continue

        # Segment is valid
        device_idx = len(valid_segments) % len(devices)
        device_segments[device_idx].append((sdx, segment))
        valid_segments.append((sdx, segment))

    log_message(f"Found {len(valid_segments)} valid segments out of {total_segments} total segments")

    # Process segments by device with batching
    for device_idx, all_device_segs in enumerate(device_segments):
        current_device = devices[device_idx]
        current_model = models[current_device]
        current_stream = streams[current_device]

        log_message(f"Processing {len(all_device_segs)} segments on device {current_device}")

        # Create batches for this device
        batches = []
        for i in range(0, len(all_device_segs), batch_size):
            batches.append(all_device_segs[i:min(i + batch_size, len(all_device_segs))])

        log_message(f"Created {len(batches)} batches of size {batch_size} for device {current_device}")

        # Process each batch
        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()
            batch_count += 1

            try:
                log_message(f"Processing batch {batch_idx + 1}/{len(batches)} on device {current_device}",
                            level="debug")

                # BATCH PREPARATION
                batch_prep_start = time.time()

                # Collect info for all segments in batch
                batch_indices = []
                batch_waveforms = []
                batch_info = []

                for sdx, segment in batch:
                    t1 = segment["start"]
                    t2 = segment["end"]
                    text = segment["text"]

                    # Convert time to frames
                    f1 = int(t1 * SAMPLE_RATE)
                    f2 = int(t2 * SAMPLE_RATE)

                    # Extract waveform for this segment
                    waveform = audio[:, f1:f2]

                    # Store metadata
                    batch_indices.append(sdx)
                    batch_waveforms.append(waveform)
                    batch_info.append({
                        "text": text,
                        "text_clean": "".join(segment_data[sdx]["clean_char"]),
                        "tokens": [model_dictionary.get(c, -1) for c in "".join(segment_data[sdx]["clean_char"])],
                        "start": t1,
                        "end": t2,
                        "duration": t2 - t1,
                        "orig_length": waveform.shape[1],
                    })

                # Find maximum length in batch for padding
                max_length = max(w.shape[1] for w in batch_waveforms)
                log_message(f"Batch max length: {max_length} frames", level="debug")

                # Pad all waveforms to the same length
                padded_waveforms = []
                for i, waveform in enumerate(batch_waveforms):
                    if waveform.shape[1] < max_length:
                        padding = (0, max_length - waveform.shape[1])
                        padded = torch.nn.functional.pad(waveform, padding)
                        padded_waveforms.append(padded)
                        log_message(f"Padded segment {i} from {waveform.shape[1]} to {max_length} frames",
                                    level="debug")
                    else:
                        padded_waveforms.append(waveform)

                # Stack into batch tensor
                batch_tensor = torch.cat(padded_waveforms, dim=0).to(current_device, non_blocking=True)

                batch_prep_time = time.time() - batch_prep_start
                batch_prep_time_total += batch_prep_time

                # MODEL INFERENCE (GPU)
                with torch.cuda.stream(current_stream):
                    # Start model inference timing
                    model_inference_start = time.time()

                    log_message(f"Running model on batch tensor of shape {batch_tensor.shape}", level="debug")

                    # Handle min length for wav2vec models
                    if batch_tensor.shape[1] < 400:
                        lengths = torch.tensor([batch_tensor.shape[1]] * batch_tensor.shape[0]).to(current_device)
                        batch_tensor = torch.nn.functional.pad(batch_tensor, (0, 400 - batch_tensor.shape[1]))
                        log_message(f"Padded batch to minimum length of 400", level="debug")
                    else:
                        lengths = None

                    with torch.inference_mode():
                        try:
                            if model_type == "torchaudio":
                                batch_emissions, _ = current_model(batch_tensor, lengths=lengths)
                            elif model_type == "huggingface":
                                batch_emissions = current_model(batch_tensor).logits
                            else:
                                raise NotImplementedError(f"Align model of type {model_type} not supported.")

                            # Log softmax on GPU
                            batch_emissions = torch.log_softmax(batch_emissions, dim=-1)

                        except Exception as e:
                            log_message(f"Error during batch inference: {str(e)}", level="error")
                            # Handle the error, but continue with other batches
                            for sdx, segment in batch:
                                aligned_seg = {
                                    "start": segment["start"],
                                    "end": segment["end"],
                                    "text": segment["text"],
                                    "words": [],
                                    "chars": None,
                                }
                                if return_char_alignments:
                                    aligned_seg["chars"] = []
                                aligned_segments.append(aligned_seg)
                            continue

                    # Sync to get accurate timing
                    current_stream.synchronize()
                    model_inference_time = time.time() - model_inference_start
                    model_inference_time_total += model_inference_time

                    log_message(f"Batch inference complete. Emissions shape: {batch_emissions.shape}", level="debug")

                # PROCESS EACH SEGMENT'S EMISSIONS (CPU)
                # Move all emissions to CPU at once for better transfer efficiency
                batch_emissions_cpu = batch_emissions.cpu()

                # Process each segment in the batch
                for i, (sdx, segment) in enumerate(batch):
                    try:
                        emission = batch_emissions_cpu[i]
                        segment_info = batch_info[i]

                        # Get tokens and info for this segment
                        tokens = segment_info["tokens"]
                        text = segment_info["text"]
                        text_clean = segment_info["text_clean"]
                        t1 = segment_info["start"]
                        t2 = segment_info["end"]
                        duration = segment_info["duration"]

                        # TRELLIS COMPUTATION
                        trellis_start = time.time()
                        trellis = get_trellis(emission, tokens, blank_id)
                        trellis_time = time.time() - trellis_start
                        trellis_time_total += trellis_time

                        # BACKTRACKING
                        backtrack_start = time.time()
                        path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)
                        backtrack_time = time.time() - backtrack_start
                        backtrack_time_total += backtrack_time

                        if path is None:
                            log_message(f'Failed to align segment ("{text}"): '
                                        f'backtrack failed, resorting to original...', level="warning")
                            aligned_segments.append({
                                "start": t1,
                                "end": t2,
                                "text": text,
                                "words": [],
                                "chars": None,
                            })
                            continue

                        # POSTPROCESSING
                        postprocess_start = time.time()

                        # Merge repeats and get character segments
                        char_segments = merge_repeats(path, text_clean)

                        # Calculate ratio for this segment
                        ratio = duration * batch_tensor.shape[1] / (trellis.size(0) - 1)

                        # Assign timestamps to aligned characters
                        char_segments_arr = []
                        word_idx = 0
                        for cdx, char in enumerate(text):
                            start, end, score = None, None, None
                            if cdx in segment_data[sdx]["clean_cdx"]:
                                char_seg = char_segments[segment_data[sdx]["clean_cdx"].index(cdx)]
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
                            elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                                word_idx += 1

                        char_segments_arr = pd.DataFrame(char_segments_arr)

                        # Process sentences
                        aligned_subsegments = []

                        # Assign sentence_idx to each character index
                        char_segments_arr["sentence-idx"] = None
                        for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
                            curr_chars = char_segments_arr.loc[
                                (char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
                            char_segments_arr.loc[
                                (char_segments_arr.index >= sstart) & (
                                            char_segments_arr.index <= send), "sentence-idx"] = sdx2

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

                                # Don't use space character for alignment
                                word_chars = word_chars[word_chars["char"] != " "]

                                word_start = word_chars["start"].min()
                                word_end = word_chars["end"].max()
                                word_score = round(word_chars["score"].mean(), 3)

                                # -1 indicates unalignable
                                word_segment = {"word": word_text}

                                if not np.isnan(word_start):
                                    word_segment["start"] = word_start
                                if not np.isnan(word_end):
                                    word_segment["end"] = word_end
                                if not np.isnan(word_score):
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
                                curr_chars = [{key: val for key, val in char.items() if val != -1} for char in
                                              curr_chars]
                                aligned_subsegments[-1]["chars"] = curr_chars

                        # Final processing
                        aligned_subsegments = pd.DataFrame(aligned_subsegments)
                        aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"],
                                                                        method=interpolate_method)
                        aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"],
                                                                      method=interpolate_method)

                        # Concatenate sentences with same timestamps
                        agg_dict = {"text": " ".join, "words": "sum"}
                        if model_lang in LANGUAGES_WITHOUT_SPACES:
                            agg_dict["text"] = "".join
                        if return_char_alignments:
                            agg_dict["chars"] = "sum"

                        aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(
                            agg_dict)
                        aligned_subsegments = aligned_subsegments.to_dict('records')
                        aligned_segments += aligned_subsegments

                        postprocess_time = time.time() - postprocess_start
                        postprocess_time_total += postprocess_time

                    except Exception as e:
                        log_message(f'Error processing segment {sdx} in batch: {str(e)}', level="error")
                        aligned_segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"],
                            "words": [],
                            "chars": None,
                        })

                batch_time = time.time() - batch_start_time
                log_message(
                    f"Batch {batch_idx + 1}/{len(batches)} on device {current_device} completed in {batch_time:.4f}s "
                    f"(prep: {batch_prep_time:.4f}s, inference: {model_inference_time:.4f}s)", level="debug")

            except Exception as e:
                log_message(f"Error processing batch {batch_idx + 1}/{len(batches)}: {str(e)}", level="error")
                # Handle batch error by falling back to individual segments
                for sdx, segment in batch:
                    aligned_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "words": [],
                        "chars": None,
                    })

        # Synchronize after processing all batches for this device
        current_stream.synchronize()

    alignment_time = time.time() - start_time_alignment

    # Create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    total_time = time.time() - start_time_total

    # Log overall timing statistics
    log_message("Performance Metrics:", level="info")
    log_message(f"Total execution time: {total_time:.4f} seconds", level="info")
    log_message(f"Audio preparation: {audio_prep_time:.4f} seconds ({audio_prep_time / total_time * 100:.2f}%)",
                level="info")
    log_message(f"Preprocessing: {preprocess_time:.4f} seconds ({preprocess_time / total_time * 100:.2f}%)",
                level="info")
    log_message(f"Alignment: {alignment_time:.4f} seconds ({alignment_time / total_time * 100:.2f}%)", level="info")
    log_message(
        f"  - Batch preparation: {batch_prep_time_total:.4f} seconds ({batch_prep_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Model inference: {model_inference_time_total:.4f} seconds ({model_inference_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Trellis computation: {trellis_time_total:.4f} seconds ({trellis_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Backtracking: {backtrack_time_total:.4f} seconds ({backtrack_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Postprocessing: {postprocess_time_total:.4f} seconds ({postprocess_time_total / total_time * 100:.2f}%)",
        level="info")

    if multi_device:
        log_message(f"Using {len(devices)} devices, {batch_count} batches of size {batch_size}, "
                    f"average time per segment: {alignment_time / total_segments:.4f} seconds",
                    level="info")

    log_message(f"Alignment completed. Processed {len(aligned_segments)} segments with {len(word_segments)} words.",
                level="info")
    return {"segments": aligned_segments, "word_segments": word_segments}

# so far best version
def align(
        transcript: Iterable[SingleSegment],
        model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
        align_model_metadata: dict,
        audio: Union[str, np.ndarray, torch.Tensor],
        device: Union[str, List[str]],
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
        combined_progress: bool = False,
        logger=None,
        batch_size: int = 1,  # Optional parameter for future batching
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    Optimized version with reduced GPU synchronization and faster backtracking.
    Supports multi-GPU processing when device is a list of device strings.
    """
    import time
    start_time_total = time.time()

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

    # Check if we're using multiple devices
    multi_device = isinstance(device, list)
    devices = device if multi_device else [device]
    models = model if multi_device else {devices[0]: model}

    log_message(f"Using {'multiple devices' if multi_device else 'single device'}: {devices}")

    # Create CUDA streams for each device
    streams = {dev: torch.cuda.Stream(device=dev) for dev in devices}

    # Time loading and preparing audio
    start_time_audio = time.time()
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    audio_prep_time = time.time() - start_time_audio
    log_message(f"Audio preparation took {audio_prep_time:.4f} seconds")

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    start_time_preprocess = time.time()
    transcript_list = list(transcript)  # Convert to list if it's an iterator
    total_segments = len(transcript_list)

    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript_list):
        # Progress reporting
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            log_message(f"Preprocessing Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
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

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd.lower()]):
                clean_wdx.append(wdx)
            else:
                # index for placeholder
                clean_wdx.append(wdx)

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }

    preprocess_time = time.time() - start_time_preprocess
    log_message(f"Preprocessing took {preprocess_time:.4f} seconds for {total_segments} segments")

    aligned_segments: List[SingleAlignedSegment] = []

    # Time for the main alignment loop
    start_time_alignment = time.time()
    model_inference_time_total = 0
    trellis_time_total = 0
    backtrack_time_total = 0
    postprocess_time_total = 0

    # Find blank_id once
    blank_id = 0
    for char, code in model_dictionary.items():
        if char == '[pad]' or char == '<pad>':
            blank_id = code

    # Group segments by device to reduce context switching
    device_segments = [[] for _ in range(len(devices))]
    for sdx, segment in enumerate(transcript_list):
        device_idx = sdx % len(devices)
        device_segments[device_idx].append((sdx, segment))

    # Process segments by device to reduce switching
    for device_idx, device_segs in enumerate(device_segments):
        current_device = devices[device_idx]
        current_model = models[current_device]
        current_stream = streams[current_device]

        # Process all segments for this device
        for segment_pair in device_segs:
            sdx, segment = segment_pair
            segment_start_time = time.time()

            if print_progress:
                progress = ((sdx + 1) / total_segments) * 100
                log_message(f"Alignment Progress: {progress:.2f}% (using device {current_device})...")

            t1 = segment["start"]
            t2 = segment["end"]
            text = segment["text"]

            aligned_seg: SingleAlignedSegment = {
                "start": t1,
                "end": t2,
                "text": text,
                "words": [],
                "chars": None,
            }

            if return_char_alignments:
                aligned_seg["chars"] = []

            # Early validation checks (keeping CPU-only checks outside the stream for better overlap)
            if len(segment_data[sdx]["clean_char"]) == 0:
                log_message(f'Failed to align segment ("{segment["text"]}"): '
                            f'no characters in this segment found in model dictionary, '
                            f'resorting to original...', level="warning")
                aligned_segments.append(aligned_seg)
                continue

            if t1 >= MAX_DURATION:
                log_message(f'Failed to align segment ("{segment["text"]}"): '
                            f'original start time longer than audio duration, skipping...', level="warning")
                aligned_segments.append(aligned_seg)
                continue

            text_clean = "".join(segment_data[sdx]["clean_char"])
            tokens = [model_dictionary.get(c, -1) for c in text_clean]

            f1 = int(t1 * SAMPLE_RATE)
            f2 = int(t2 * SAMPLE_RATE)

            # Use the current CUDA stream for GPU operations
            with torch.cuda.stream(current_stream):
                # Move audio segment to device (non-blocking for better overlap)
                audio_to_device_start = time.time()
                waveform_segment = audio[:, f1:f2].to(current_device, non_blocking=True)
                # Ensure the transfer is complete before proceeding
                current_stream.synchronize()
                audio_to_device_time = time.time() - audio_to_device_start

                # Handle the minimum input length for wav2vec2 models
                if waveform_segment.shape[-1] < 400:
                    lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(current_device)
                    waveform_segment = torch.nn.functional.pad(
                        waveform_segment, (0, 400 - waveform_segment.shape[-1])
                    )
                else:
                    lengths = None

                # Model inference - keep on GPU
                model_inference_start = time.time()
                with torch.inference_mode():
                    try:
                        if model_type == "torchaudio":
                            emissions, _ = current_model(waveform_segment, lengths=lengths)
                        elif model_type == "huggingface":
                            emissions = current_model(waveform_segment).logits
                        else:
                            raise NotImplementedError(f"Align model of type {model_type} not supported.")

                        # Calculate log_softmax on GPU to reduce transfers
                        emissions = torch.log_softmax(emissions, dim=-1)

                        # Store waveform size for later
                        waveform_size = waveform_segment.size(0)
                        trellis_size_0 = None  # Will be set after trellis computation
                    except Exception as e:
                        log_message(f'Failed to get model predictions for segment ("{segment["text"]}"): {str(e)}',
                                    level="error")
                        aligned_segments.append(aligned_seg)
                        continue

                # Complete model inference timing
                current_stream.synchronize()  # Make sure inference is complete for timing
                model_inference_time = time.time() - model_inference_start
                model_inference_time_total += model_inference_time

                # CPU operations can start once we have the emissions
                emission = emissions[0].cpu()

            # CPU operations (outside the stream context for better overlap)
            # Time trellis computation
            trellis_start = time.time()
            trellis = get_trellis(emission, tokens, blank_id)
            trellis_time = time.time() - trellis_start
            trellis_time_total += trellis_time

            # Store trellis size for ratio calculation
            trellis_size_0 = trellis.size(0)

            # Optimized backtracking (this is a CPU operation)
            backtrack_start = time.time()
            # Use optimized backtrack function
            path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)
            backtrack_time = time.time() - backtrack_start
            backtrack_time_total += backtrack_time

            if path is None:
                log_message(f'Failed to align segment ("{segment["text"]}"): '
                            f'backtrack failed, resorting to original...', level="warning")
                aligned_segments.append(aligned_seg)
                continue

            # Time postprocessing
            postprocess_start = time.time()
            char_segments = merge_repeats(path, text_clean)

            duration = t2 - t1
            ratio = duration * waveform_size / (trellis_size_0 - 1)

            # assign timestamps to aligned characters
            char_segments_arr = []
            word_idx = 0
            for cdx, char in enumerate(text):
                start, end, score = None, None, None
                if cdx in segment_data[sdx]["clean_cdx"]:
                    char_seg = char_segments[segment_data[sdx]["clean_cdx"].index(cdx)]
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

                # increment word_idx, nltk word tokenization would probably be more robust here
                if model_lang in LANGUAGES_WITHOUT_SPACES:
                    word_idx += 1
                elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                    word_idx += 1

            char_segments_arr = pd.DataFrame(char_segments_arr)

            aligned_subsegments = []
            # assign sentence_idx to each character index
            try:
                # assign sentence_idx to each character index
                char_segments_arr["sentence-idx"] = None
                for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
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

                        word_start = word_chars["start"].min()
                        word_end = word_chars["end"].max()
                        word_score = round(word_chars["score"].mean(), 3)

                        # -1 indicates unalignable
                        word_segment = {"word": word_text}

                        if not np.isnan(word_start):
                            word_segment["start"] = word_start
                        if not np.isnan(word_end):
                            word_segment["end"] = word_end
                        if not np.isnan(word_score):
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
                aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
                aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
                # concatenate sentences with same timestamps
                agg_dict = {"text": " ".join, "words": "sum"}
                if model_lang in LANGUAGES_WITHOUT_SPACES:
                    agg_dict["text"] = "".join
                if return_char_alignments:
                    agg_dict["chars"] = "sum"
                aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
                aligned_subsegments = aligned_subsegments.to_dict('records')
                aligned_segments += aligned_subsegments
            except Exception as e:
                log_message(f'Error processing aligned characters for segment ("{segment["text"]}"): {str(e)}',
                            level="error")
                aligned_segments.append(aligned_seg)
                continue

            postprocess_time = time.time() - postprocess_start
            postprocess_time_total += postprocess_time

            segment_total_time = time.time() - segment_start_time
            if print_progress:
                log_message(f"Segment {sdx + 1}/{total_segments} timing: "
                            f"Total: {segment_total_time:.4f}s, "
                            f"Audio to device: {audio_to_device_time:.4f}s, "
                            f"Model inference: {model_inference_time:.4f}s, "
                            f"Trellis: {trellis_time:.4f}s, "
                            f"Backtrack: {backtrack_time:.4f}s, "
                            f"Postprocess: {postprocess_time:.4f}s")

        # Synchronize the stream after processing all segments for this device
        current_stream.synchronize()

    alignment_time = time.time() - start_time_alignment

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    total_time = time.time() - start_time_total

    # Log overall timing statistics
    log_message("Performance Metrics:", level="info")
    log_message(f"Total execution time: {total_time:.4f} seconds", level="info")
    log_message(f"Audio preparation: {audio_prep_time:.4f} seconds ({audio_prep_time / total_time * 100:.2f}%)",
                level="info")
    log_message(f"Preprocessing: {preprocess_time:.4f} seconds ({preprocess_time / total_time * 100:.2f}%)",
                level="info")
    log_message(f"Alignment: {alignment_time:.4f} seconds ({alignment_time / total_time * 100:.2f}%)", level="info")
    log_message(
        f"  - Model inference: {model_inference_time_total:.4f} seconds ({model_inference_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Trellis computation: {trellis_time_total:.4f} seconds ({trellis_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Backtracking: {backtrack_time_total:.4f} seconds ({backtrack_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Postprocessing: {postprocess_time_total:.4f} seconds ({postprocess_time_total / total_time * 100:.2f}%)",
        level="info")

    if multi_device:
        log_message(
            f"Using {len(devices)} devices, average time per segment: {alignment_time / total_segments:.4f} seconds",
            level="info")

    log_message(f"Alignment completed. Processed {len(aligned_segments)} segments with {len(word_segments)} words.",
                level="info")
    return {"segments": aligned_segments, "word_segments": word_segments}


def align1(
        transcript: Iterable[SingleSegment],
        model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
        align_model_metadata: dict,
        audio: Union[str, np.ndarray, torch.Tensor],
        device: Union[str, List[str]],
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        print_progress: bool = False,
        combined_progress: bool = False,
        logger=None,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    Now supports multi-GPU processing when device is a list of device strings.
    """
    import time
    start_time_total = time.time()

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

    # Check if we're using multiple devices
    multi_device = isinstance(device, list)
    devices = device if multi_device else [device]
    models = model if multi_device else {devices[0]: model}

    log_message(f"Using {'multiple devices' if multi_device else 'single device'}: {devices}")

    # Time loading and preparing audio
    start_time_audio = time.time()
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    audio_prep_time = time.time() - start_time_audio
    log_message(f"Audio preparation took {audio_prep_time:.4f} seconds")

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    start_time_preprocess = time.time()
    transcript_list = list(transcript)  # Convert to list if it's an iterator
    total_segments = len(transcript_list)

    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript_list):
        # Progress reporting
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            log_message(f"Preprocessing Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
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

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd.lower()]):
                clean_wdx.append(wdx)
            else:
                # index for placeholder
                clean_wdx.append(wdx)

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }

    preprocess_time = time.time() - start_time_preprocess
    log_message(f"Preprocessing took {preprocess_time:.4f} seconds for {total_segments} segments")

    aligned_segments: List[SingleAlignedSegment] = []

    # Time for the main alignment loop
    start_time_alignment = time.time()
    model_inference_time_total = 0
    trellis_time_total = 0
    backtrack_time_total = 0
    postprocess_time_total = 0

    # Process segments in alternating fashion across GPUs
    num_devices = len(devices)
    for sdx, segment in enumerate(transcript_list):
        segment_start_time = time.time()

        # Determine which device to use for this segment
        device_idx = sdx % num_devices
        current_device = devices[device_idx]
        current_model = models[current_device]

        if print_progress:
            progress = ((sdx + 1) / total_segments) * 100
            log_message(f"Alignment Progress: {progress:.2f}% (using device {current_device})...")

        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": None,
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment_data[sdx]["clean_char"]) == 0:
            log_message(f'Failed to align segment ("{segment["text"]}"): '
                        f'no characters in this segment found in model dictionary, '
                        f'resorting to original...', level="warning")
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            log_message(f'Failed to align segment ("{segment["text"]}"): '
                        f'original start time longer than audio duration, skipping...', level="warning")
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])
        tokens = [model_dictionary.get(c, -1) for c in text_clean]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # Move audio segment to the appropriate device
        audio_to_device_start = time.time()
        waveform_segment = audio[:, f1:f2].to(current_device)
        audio_to_device_time = time.time() - audio_to_device_start

        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(current_device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None

        # Time model inference
        model_inference_start = time.time()
        with torch.inference_mode():
            try:
                if model_type == "torchaudio":
                    emissions, _ = current_model(waveform_segment, lengths=lengths)
                elif model_type == "huggingface":
                    emissions = current_model(waveform_segment).logits
                else:
                    raise NotImplementedError(f"Align model of type {model_type} not supported.")
                emissions = torch.log_softmax(emissions, dim=-1)
            except Exception as e:
                log_message(f'Failed to get model predictions for segment ("{segment["text"]}"): {str(e)}',
                            level="error")
                aligned_segments.append(aligned_seg)
                continue
        model_inference_time = time.time() - model_inference_start
        model_inference_time_total += model_inference_time

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        # Time trellis computation
        trellis_start = time.time()
        trellis = get_trellis(emission, tokens, blank_id)
        trellis_time = time.time() - trellis_start
        trellis_time_total += trellis_time

        # Time backtracking
        backtrack_start = time.time()
        path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)
        backtrack_time = time.time() - backtrack_start
        backtrack_time_total += backtrack_time

        if path is None:
            log_message(f'Failed to align segment ("{segment["text"]}"): '
                        f'backtrack failed, resorting to original...', level="warning")

            aligned_segments.append(aligned_seg)
            continue

        # Time postprocessing
        postprocess_start = time.time()
        char_segments = merge_repeats(path, text_clean)

        duration = t2 - t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment_data[sdx]["clean_cdx"]:
                char_seg = char_segments[segment_data[sdx]["clean_cdx"].index(cdx)]
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

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                word_idx += 1

        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # assign sentence_idx to each character index
        try:
            # assign sentence_idx to each character index
            char_segments_arr["sentence-idx"] = None
            for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
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

                    word_start = word_chars["start"].min()
                    word_end = word_chars["end"].max()
                    word_score = round(word_chars["score"].mean(), 3)

                    # -1 indicates unalignable
                    word_segment = {"word": word_text}

                    if not np.isnan(word_start):
                        word_segment["start"] = word_start
                    if not np.isnan(word_end):
                        word_segment["end"] = word_end
                    if not np.isnan(word_score):
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
            aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
            aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
            # concatenate sentences with same timestamps
            agg_dict = {"text": " ".join, "words": "sum"}
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                agg_dict["text"] = "".join
            if return_char_alignments:
                agg_dict["chars"] = "sum"
            aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
            aligned_subsegments = aligned_subsegments.to_dict('records')
            aligned_segments += aligned_subsegments
        except Exception as e:
            log_message(f'Error processing aligned characters for segment ("{segment["text"]}"): {str(e)}',
                        level="error")
            aligned_segments.append(aligned_seg)
            continue

        postprocess_time = time.time() - postprocess_start
        postprocess_time_total += postprocess_time

        segment_total_time = time.time() - segment_start_time
        if print_progress:
            log_message(f"Segment {sdx + 1}/{total_segments} timing: "
                        f"Total: {segment_total_time:.4f}s, "
                        f"Audio to device: {audio_to_device_time:.4f}s, "
                        f"Model inference: {model_inference_time:.4f}s, "
                        f"Trellis: {trellis_time:.4f}s, "
                        f"Backtrack: {backtrack_time:.4f}s, "
                        f"Postprocess: {postprocess_time:.4f}s")

    alignment_time = time.time() - start_time_alignment

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    total_time = time.time() - start_time_total

    # Log overall timing statistics
    log_message("Performance Metrics:", level="info")
    log_message(f"Total execution time: {total_time:.4f} seconds", level="info")
    log_message(f"Audio preparation: {audio_prep_time:.4f} seconds ({audio_prep_time / total_time * 100:.2f}%)",
                level="info")
    log_message(f"Preprocessing: {preprocess_time:.4f} seconds ({preprocess_time / total_time * 100:.2f}%)",
                level="info")
    log_message(f"Alignment: {alignment_time:.4f} seconds ({alignment_time / total_time * 100:.2f}%)", level="info")
    log_message(
        f"  - Model inference: {model_inference_time_total:.4f} seconds ({model_inference_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Trellis computation: {trellis_time_total:.4f} seconds ({trellis_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Backtracking: {backtrack_time_total:.4f} seconds ({backtrack_time_total / total_time * 100:.2f}%)",
        level="info")
    log_message(
        f"  - Postprocessing: {postprocess_time_total:.4f} seconds ({postprocess_time_total / total_time * 100:.2f}%)",
        level="info")

    if multi_device:
        log_message(
            f"Using {len(devices)} devices, average time per segment: {alignment_time / total_segments:.4f} seconds",
            level="info")

    log_message(f"Alignment completed. Processed {len(aligned_segments)} segments with {len(word_segments)} words.",
                level="info")
    return {"segments": aligned_segments, "word_segments": word_segments}


def align0(
    transcript: Iterable[SingleSegment],
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: Union[str, List[str]],
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    logger = None,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    Now supports multi-GPU processing when device is a list of device strings.
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
    
    # Check if we're using multiple devices
    multi_device = isinstance(device, list)
    devices = device if multi_device else [device]
    models = model if multi_device else {devices[0]: model}
    
    log_message(f"Using {'multiple devices' if multi_device else 'single device'}: {devices}")

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

    # 1. Preprocess to keep only characters in dictionary
    transcript_list = list(transcript)  # Convert to list if it's an iterator
    total_segments = len(transcript_list)
    
    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript_list):
        # Progress reporting
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            log_message(f"Preprocessing Progress: {percent_complete:.2f}%...")
            
        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
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

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any([c in model_dictionary.keys() for c in wrd.lower()]):
                clean_wdx.append(wdx)
            else:
                # index for placeholder
                clean_wdx.append(wdx)
                
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }
            
    aligned_segments: List[SingleAlignedSegment] = []
    
    # Process segments in alternating fashion across GPUs
    num_devices = len(devices)
    for sdx, segment in enumerate(transcript_list):
        # Determine which device to use for this segment
        device_idx = sdx % num_devices
        current_device = devices[device_idx]
        current_model = models[current_device]
        
        if print_progress:
            progress = ((sdx + 1) / total_segments) * 100
            log_message(f"Alignment Progress: {progress:.2f}% (using device {current_device})...")
        
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": None,
        }

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment_data[sdx]["clean_char"]) == 0:
            log_message(f'Failed to align segment ("{segment["text"]}"): '
                        f'no characters in this segment found in model dictionary, '
                        f'resorting to original...', level="warning")
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            log_message(f'Failed to align segment ("{segment["text"]}"): '
                        f'original start time longer than audio duration, skipping...', level="warning")
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])
        tokens = [model_dictionary.get(c, -1) for c in text_clean]

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # Move audio segment to the appropriate device
        waveform_segment = audio[:, f1:f2].to(current_device)
        
        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(current_device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None
            
        with torch.inference_mode():
            try:
                if model_type == "torchaudio":
                    emissions, _ = current_model(waveform_segment, lengths=lengths)
                elif model_type == "huggingface":
                    emissions = current_model(waveform_segment).logits
                else:
                    raise NotImplementedError(f"Align model of type {model_type} not supported.")
                emissions = torch.log_softmax(emissions, dim=-1)
            except Exception as e:
                log_message(f'Failed to get model predictions for segment ("{segment["text"]}"): {str(e)}', level="error")
                aligned_segments.append(aligned_seg)
                continue

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)

        if path is None:
            log_message(f'Failed to align segment ("{segment["text"]}"): '
                        f'backtrack failed, resorting to original...', level="warning")

            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 - t1
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment_data[sdx]["clean_cdx"]:
                char_seg = char_segments[segment_data[sdx]["clean_cdx"].index(cdx)]
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

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx+1] == " ":
                word_idx += 1
            
        char_segments_arr = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        # assign sentence_idx to each character index
        try:
            # assign sentence_idx to each character index
            char_segments_arr["sentence-idx"] = None
            for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
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

                    word_start = word_chars["start"].min()
                    word_end = word_chars["end"].max()
                    word_score = round(word_chars["score"].mean(), 3)

                    # -1 indicates unalignable
                    word_segment = {"word": word_text}

                    if not np.isnan(word_start):
                        word_segment["start"] = word_start
                    if not np.isnan(word_end):
                        word_segment["end"] = word_end
                    if not np.isnan(word_score):
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
            aligned_subsegments["start"] = interpolate_nans(aligned_subsegments["start"], method=interpolate_method)
            aligned_subsegments["end"] = interpolate_nans(aligned_subsegments["end"], method=interpolate_method)
            # concatenate sentences with same timestamps
            agg_dict = {"text": " ".join, "words": "sum"}
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                agg_dict["text"] = "".join
            if return_char_alignments:
                agg_dict["chars"] = "sum"
            aligned_subsegments = aligned_subsegments.groupby(["start", "end"], as_index=False).agg(agg_dict)
            aligned_subsegments = aligned_subsegments.to_dict('records')
            aligned_segments += aligned_subsegments
        except Exception as e:
            log_message(f'Error processing aligned characters for segment ("{segment["text"]}"): {str(e)}',
                        level="error")
            aligned_segments.append(aligned_seg)
            continue

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    log_message(f"Alignment completed. Processed {len(aligned_segments)} segments with {len(word_segments)} words.",
                level="info")
    return {"segments": aligned_segments, "word_segments": word_segments}


@nb.jit(nopython=True)
def get_trellis_numba(emission_np, tokens_np, blank_id=0):
    """JIT-compiled trellis computation."""
    num_frame = emission_np.shape[0]
    num_tokens = len(tokens_np)

    trellis = np.zeros((num_frame, num_tokens), dtype=np.float32)

    # Initialize first row/column
    trellis[1:, 0] = np.cumsum(emission_np[1:, blank_id])
    trellis[0, 1:] = -np.inf
    if num_frame >= num_tokens:
        trellis[-num_tokens + 1:, 0] = np.inf

    # Dynamic programming
    for t in range(num_frame - 1):
        for j in range(1, num_tokens):
            # Score for staying
            stay_score = trellis[t, j] + emission_np[t, blank_id]

            # Score for changing
            change_score = trellis[t, j - 1]
            if tokens_np[j] == -1:  # Wildcard
                # Find max emission excluding blank
                valid_emissions = emission_np[t].copy()
                valid_emissions[blank_id] = -np.inf
                token_emission = np.max(valid_emissions)
            else:
                token_emission = emission_np[t, tokens_np[j]]

            change_score += token_emission

            # Take maximum
            trellis[t + 1, j] = max(stay_score, change_score)

    return trellis

# The following functions are preserved from the original code
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


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


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


@nb.jit(nopython=True, parallel=True)
def backtrack_beam(trellis, emission, tokens, blank_id=0, beam_width=5):
    """Standard CTC beam search backtracking implementation."""
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


# def backtrack_beam_attempt(trellis, emission, tokens, blank_id=0, beam_width=5):
#     """Optimized CTC beam search backtracking implementation."""
#     T, J = trellis.size(0) - 1, trellis.size(1) - 1
#
#     # Pre-compute emission values to avoid redundant computation
#     emission_cpu = emission.cpu().numpy()
#     blank_emission = emission_cpu[:, blank_id]
#
#     # Initial beam
#     init_state = BeamState(
#         token_index=J,
#         time_index=T,
#         score=trellis[T, J].item(),  # Convert to Python float (faster)
#         path=[Point(J, T, math.exp(emission_cpu[T, blank_id]))]  # Pre-compute exp
#     )
#
#     beams = [init_state]
#
#     while beams and beams[0].token_index > 0:
#         next_beams = []
#
#         for beam in beams:
#             t, j = beam.time_index, beam.token_index
#
#             if t <= 0:
#                 continue
#
#             # Get scores once (avoid repeated tensor indexing)
#             stay_score = trellis[t - 1, j].item()
#             change_score = trellis[t - 1, j - 1].item() if j > 0 else float('-inf')
#
#             # Handle token emissions efficiently
#             token_j = tokens[j] if j < len(tokens) else -1
#             if token_j == -1:
#                 # For wildcard, find max emission excluding blank
#                 valid_emissions = emission_cpu[t - 1].copy()
#                 valid_emissions[blank_id] = float('-inf')
#                 p_change = valid_emissions.max()
#             else:
#                 p_change = emission_cpu[t - 1, token_j]
#
#             # Stay path - avoid unnecessary deep copies
#             if not math.isinf(stay_score):
#                 new_path = list(beam.path)  # Shallow copy is much faster
#                 new_path.append(Point(j, t - 1, math.exp(blank_emission[t - 1])))
#
#                 next_beams.append(BeamState(
#                     token_index=j,
#                     time_index=t - 1,
#                     score=stay_score,
#                     path=new_path
#                 ))
#
#             # Change path
#             if j > 0 and not math.isinf(change_score):
#                 new_path = list(beam.path)
#                 new_path.append(Point(j - 1, t - 1, math.exp(p_change)))
#
#                 next_beams.append(BeamState(
#                     token_index=j - 1,
#                     time_index=t - 1,
#                     score=change_score,
#                     path=new_path
#                 ))
#
#         # Use key function for faster sorting
#         if next_beams:
#             next_beams.sort(key=lambda x: x.score, reverse=True)
#             beams = next_beams[:beam_width]
#         else:
#             break
#
#     if not beams:
#         return None
#
#     # Process best beam
#     best_beam = beams[0]
#     t = best_beam.time_index
#     j = best_beam.token_index
#
#     # Complete path to t=0 efficiently (single loop)
#     while t > 0:
#         t -= 1
#         best_beam.path.append(Point(j, t, math.exp(blank_emission[t])))
#
#     return best_beam.path[::-1]

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