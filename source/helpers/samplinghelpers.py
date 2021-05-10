# Copyright 2021 Tristan Behrens.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import note_seq
import random
from source.helpers.noteseqhelpers import (
    empty_note_sequence,
    NOTE_LENGTH_16TH_120BPM,
    BAR_LENGTH_120BPM
)


def render_token_sequence(token_sequence, use_program=True, use_drums=True):
    note_sequence = token_sequence_to_note_sequence(token_sequence, use_program=use_program, use_drums=use_drums)
    synth = note_seq.midi_synth.fluidsynth
    note_seq.plot_sequence(note_sequence)
    note_seq.play_sequence(note_sequence, synth)


def print_token_sequence(token_sequence, priming_samples_number=None):

    if isinstance(token_sequence, str):
        token_sequence = token_sequence.split()
    assert isinstance(token_sequence, list)

    indent_level = 0
    result = ""
    for token_index, token in enumerate(token_sequence):

        if priming_samples_number is not None:
            if token_index < priming_samples_number:
                first_character = "P "
            else:
                first_character = "  "
        else:
            first_character = ""

        if token in ["PIECE_END", "TRACK_END", "BAR_END"]:
            indent_level -= 1

        result += first_character + f"{token_index:04d} " + "  " * indent_level + token + "\n"

        if token in ["PIECE_START", "TRACK_START", "BAR_START"]:
            indent_level += 1

    print(result)


def get_priming_token_sequence(data_path, stop_on_track_end=None, stop_after_n_tokens=None, return_original=False):

    # Get a random token sequence from the file.
    lines = open(data_path, "r").readlines()
    token_sequence = random.choice(lines)

    result_tokens = []
    track_end_index = 0
    for token_index, token in enumerate(token_sequence.split()):
        result_tokens += [token]

        if stop_on_track_end == track_end_index and token == "TRACK_END":
            break

        if token == "TRACK_END":
            track_end_index += 1

        if stop_after_n_tokens != 0 and token_index + 1 == stop_after_n_tokens:
            break

    result = " ".join(result_tokens)
    if not return_original:
        return result
    else:
        return result, token_sequence


def generate(model, tokenizer, token_sequence):

    # Map token sequence to ids.
    input_ids = tokenizer.encode(token_sequence, return_tensors="pt")

    generated_sequence = model.generate(
        input_ids,
        #min_length=200,
        max_length=1000,
        temperature=0.9,
        #pad_token_id=tokenizer.token_to_id("[PAD]"),
        #bos_token_id=tokenizer.token_to_id("PIECE_START"),
        #eos_token_id=tokenizer.token_to_id("PIECE_END"),
        #bad_words_ids=[[tokenizer.token_to_id("[PAD]")], [tokenizer.token_to_id("[MASK]")]]
    )
    generated_sequence = tokenizer.decode(generated_sequence[0])
    return generated_sequence


def token_sequence_to_note_sequence(token_sequence, use_program=True, use_drums=True):

    if isinstance(token_sequence, str):
        token_sequence = token_sequence.split()

    note_sequence = empty_note_sequence()
    current_program = 1
    current_is_drum = False
    for token_index, token in enumerate(token_sequence):

        if token == "PIECE_START":
            pass
        elif token == "PIECE_END":
            print("The end.")
            break
        elif token == "TRACK_START":
            current_bar_index = 0
            pass
        elif token == "TRACK_END":
            pass
        elif token.startswith("INST"):
            current_instrument = token.split("=")[-1]
            if current_instrument != "DRUMS" and use_program:
                current_instrument = int(current_instrument)
                current_program = int(current_instrument)
                current_is_drum = False
            if current_instrument == "DRUMS" and use_drums:
                current_instrument = 0
                current_program = 0
                current_is_drum = True
        elif token == "BAR_START":
            current_time = current_bar_index * BAR_LENGTH_120BPM
            current_notes = {}
        elif token == "BAR_END":
            current_bar_index += 1
            pass
        elif token.startswith("NOTE_ON"):
            pitch = int(token.split("=")[-1])
            note = note_sequence.notes.add()
            note.start_time = current_time
            note.end_time = current_time + 4 * NOTE_LENGTH_16TH_120BPM
            note.pitch = pitch
            note.instrument = int(current_instrument)
            note.program = current_program
            note.velocity = 80
            note.is_drum = current_is_drum
            current_notes[pitch] = note
        elif token.startswith("NOTE_OFF"):
            pitch = int(token.split("=")[-1])
            if pitch in current_notes:
                note = current_notes[pitch]
                note.end_time = current_time
        elif token.startswith("TIME_DELTA"):
            delta = float(token.split("=")[-1]) * NOTE_LENGTH_16TH_120BPM
            current_time += delta
        elif token.startswith("DENSITY="):
            pass
        elif token == "[PAD]":
            pass
        else:
            assert False, token

    return note_sequence
