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

NOTE_LENGTH_16TH_120BPM = 0.25 * 60 / 120
BAR_LENGTH_120BPM = 4.0 * 60 / 120


def set_note_sequence_tempo(note_sequence, target_tempo):

    raise_exception_on_multiple_tempos(note_sequence)

    # Find multiplier.
    current_tempo = note_sequence.tempos[0].qpm
    multiplier = current_tempo / target_tempo

    # Set tempo.
    note_sequence.tempos[0].qpm = target_tempo

    # Set total_time.
    note_sequence.total_time *= multiplier

    # Time stretching. Multiply all notes.
    for note in note_sequence.notes:
        note.start_time *= multiplier
        note.end_time *= multiplier

    # Done.
    return note_sequence


def split_note_sequence_into_bars(note_sequence, absolute_times, threshold=0.0, quantized=False):

    # We cannot handle tempo changes.
    raise_exception_on_multiple_tempos(note_sequence)

    # Get the qpm for later.
    qpm = note_sequence.tempos[0].qpm

    # Compute bar length.
    bar_length = 4 * 60.0 / qpm

    # Split the note sequence into bars.
    if not quantized:
        bars = note_sequence_to_bars(note_sequence, threshold=threshold)
    else:
        bars = note_sequence_to_bars_quantized(note_sequence)
    assert len(bars) != 0

    # Map to note sequences.
    note_sequences = bars_to_note_sequences(bars, qpm, bar_length, absolute_times)
    assert len(note_sequences) != 0

    # Done.
    return note_sequences


def note_sequence_to_bars(note_sequence, threshold):

    # Get the qpm for later.
    qpm = note_sequence.tempos[0].qpm

    # Getting the bar length from qpm.
    bar_length = 4 * 60.0 / qpm

    # Get the number of bars.
    bars_number = int(round(note_sequence.total_time / bar_length))

    # To through all bars.
    bars = []
    start_time = 0.0
    processed_notes = []
    for index in range(bars_number):

        # Done if we ran out of notes.
        if len(processed_notes) >= len(note_sequence.notes):
            break

        # Compute the end time.
        end_time = start_time + bar_length

        # Go through the notes and find the ones that fit the bar.
        notes = []
        for note in note_sequence.notes:
            if note.start_time >= start_time - threshold and note.start_time < end_time - threshold:
                notes += [note]

        # Store in result.
        bars += [notes]

        # The new start time is the end time.
        start_time = end_time

    # Done
    return bars


def note_sequence_to_bars_quantized(note_sequence, steps_per_bar=16):

    # Sort the notes.
    notes_to_process = sorted(note_sequence.notes, key=lambda note: note.quantized_start_step)

    # Go through all note in a bar-wise fashion.
    bars = []
    step_start = 0
    steps_maximum = steps_per_bar * 100
    for step_start, step_end in zip(range(0, steps_maximum, steps_per_bar), range(steps_per_bar, steps_maximum, steps_per_bar)):

        # Done. Leave the loop.
        if len(notes_to_process) == 0:
            break

        # Find all the notes that are in the bar.
        notes_found = []
        for note in notes_to_process:
            # Add note to the list if it is in the bar.
            if note.quantized_start_step >= step_start and note.quantized_start_step < step_end:
                notes_found += [note]

            # This note is not in the bar. Stop loop.
            elif note.quantized_start_step >= step_end:
                break

        # Remove those notes.
        for note in notes_found:
            notes_to_process.remove(note)

        # Append.
        bars += [notes_found]

    # Done
    return bars


def bars_to_note_sequences(bars, qpm, bar_length, absolute_times):

    # Array for the note sequences.
    note_sequences = []

    # Go through all bars.
    for bar_index, bar in enumerate(bars):
        note_sequence_bar = empty_note_sequence(qpm=qpm, total_time=bar_length)

        # Go through all notes and copy them.
        for note in bar:
            new_note = note_sequence_bar.notes.add()
            new_note.CopyFrom(note)

            # Do time shifting if absolute times are requested.
            if not absolute_times:
                new_note.start_time -= bar_index * bar_length
                new_note.end_time -= bar_index * bar_length
                new_note.quantized_start_step -= bar_index * 16
                new_note.quantized_end_step -= bar_index * 16

        # Add to list.
        note_sequences.append(note_sequence_bar)

    # Done.
    return note_sequences


def clip_quantized_steps(note_sequence, steps):
    for note in note_sequence.notes:
        if note.quantized_start_step < 0:
            note.quantized_start_step = 0
        if note.quantized_start_step >= steps:
            note.quantized_start_step = steps - 1
        if note.quantized_end_step < 1:
            note.quantized_end_step = 1
        if note.quantized_end_step > steps:
            note.quantized_end_step = steps
    return note_sequence


def empty_note_sequence(qpm=120.0, total_time=0.0):
    note_sequence = note_seq.protobuf.music_pb2.NoteSequence()
    note_sequence.tempos.add().qpm = qpm
    note_sequence.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    note_sequence.total_time = total_time
    return note_sequence


def raise_exception_on_multiple_tempos(note_sequence):
    if len(note_sequence.tempos) != 1:
        error_message = f"Too many tempos: {len(note_sequence.tempos)}"
        raise Exception(error_message)
