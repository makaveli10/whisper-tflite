import itertools
import logging
import os
import zlib
import numpy as np
import tensorflow as tf
import tokenizers
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union

from whisper_tflite.audio import decode_audio
from whisper_tflite.feature_extractor import FeatureExtractor
from whisper_tflite.tokenizer import Tokenizer
from whisper_tflite.utils import format_timestamp, get_logger


class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]



class TranscriptionOptions(NamedTuple):
    best_of: int
    patience: float
    length_penalty: float
    log_prob_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    temperatures: List[float]
    initial_prompt: Optional[str]
    prefix: Optional[str]
    suppress_blank: bool
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str


class TranscriptionInfo(NamedTuple):
    language: str
    language_probability: float
    duration: float
    all_language_probs: Optional[List[Tuple[str, float]]]


class WhisperModel:
    def __init__(
        self,
        tflite_model_path: str,
    ):
        """Initializes the Whisper model.

        Args:
          tflite_model_path: path to the whisper tflite model to use
        """
        self.logger = get_logger()


        # Create an interpreter to run the TFLite model
        self.model = tf.lite.Interpreter(tflite_model_path)

        # Allocate memory for the interpreter
        self.model.allocate_tensors()

        # Get the input and output tensors
        self.input_tensor = self.model.get_input_details()[0]['index']
        self.output_tensor = self.model.get_output_details()[0]['index']

        # self.hf_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en", predict_timestamps=True)
        self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
                "openai/whisper-tiny.en"
            )
        self.feature_extractor = FeatureExtractor()
        self.num_samples_per_token = self.feature_extractor.hop_length * 2
        self.frames_per_second = (
            self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
        )
        self.tokens_per_second = (
            self.feature_extractor.sampling_rate // self.num_samples_per_token
        )
        self.input_stride = 2
        self.time_precision = 0.02
        self.max_length = 448

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribes an input file.

        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform.
          language: The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.
          task: Task to execute (transcribe or translate).

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """
        sampling_rate = self.feature_extractor.sampling_rate

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate

        self.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )


        features = self.feature_extractor(audio)

        encoder_output = None
        all_language_probs = None

        language = "en"
        language_probability = 1


        tokenizer = Tokenizer(
            self.hf_tokenizer,
            False,
            task=task,
            language=language,
        )

        segments = self.generate_segments(features, tokenizer)

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            all_language_probs=all_language_probs,
        )

        return segments, info

    def generate_segments(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
    ) -> Iterable[Segment]:
        content_frames = features.shape[-1] - self.feature_extractor.nb_max_frames
        idx = 0
        seek = 0
        all_tokens = []
        prompt_reset_since = 0

        while seek < content_frames:
            time_offset = seek * self.feature_extractor.time_per_frame
            segment = features[:, seek : seek + self.feature_extractor.nb_max_frames]
            segment_size = min(
                self.feature_extractor.nb_max_frames, content_frames - seek
            )
            segment_duration = segment_size * self.feature_extractor.time_per_frame

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Processing segment at %s", format_timestamp(time_offset)
                )

            previous_tokens = all_tokens[prompt_reset_since:]

            self.model.set_tensor(self.input_tensor, np.expand_dims(segment, axis=0))
            self.model.invoke()

            # Get the output data from the interpreter
            result = self.model.get_tensor(self.output_tensor)


            tokens = result[0]
            tokens = tokens[0:np.where(tokens == tokenizer.eot)[0][0]+1]
            tokens = tokens[1:-1]


            previous_seek = seek
            current_segments = []

            single_timestamp_ending = (
                len(tokens) >= 2
                and tokens[-2] < tokenizer.timestamp_begin
                and tokens[-1] >= tokenizer.timestamp_begin
            )

            consecutive_timestamps = [
                i
                for i in range(len(tokens))
                if i > 0
                and tokens[i] >= tokenizer.timestamp_begin
                and tokens[i - 1] >= tokenizer.timestamp_begin
            ]
            if len(consecutive_timestamps) > 0:
                slices = list(consecutive_timestamps)
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                        sliced_tokens[0] - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                        sliced_tokens[-1] - tokenizer.timestamp_begin
                    )
                    start_time = (
                        time_offset + start_timestamp_position * self.time_precision
                    )
                    end_time = (
                        time_offset + end_timestamp_position * self.time_precision
                    )

                    current_segments.append(
                        dict(
                            seek=seek,
                            start=start_time,
                            end=end_time,
                            tokens=sliced_tokens,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_position = (
                        tokens[last_slice - 1] - tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_position * self.input_stride

            else:
                duration = segment_duration
                timestamps = [
                    token for token in tokens if token >= tokenizer.timestamp_begin
                ]
                if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
                    last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                    duration = last_timestamp_position * self.time_precision

                current_segments.append(
                    dict(
                        seek=seek,
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                    )
                )

                seek += segment_size

            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.tokenizer.decode(tokens)

                if segment["start"] == segment["end"] or not text.strip():
                    continue

                all_tokens.extend(tokens)
                idx += 1

                yield Segment(
                    id=idx,
                    seek=seek,
                    start=segment["start"],
                    end=segment["end"],
                    text=text,
                    tokens=tokens
                )