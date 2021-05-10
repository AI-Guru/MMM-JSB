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

import music21
from music21 import corpus
from source import logging
from source.preprocess.preprocessutilities import events_to_events_data

logger = logging.create_logger("music21jsb")


def preprocess_music21():

    logger.info("Loading songs...")
    songs = list(corpus.chorales.Iterator())
    logger.info(f"Got {len(songs)} songs.")

    split_index = int(0.8 * len(songs))
    songs_train = songs[:split_index]
    songs_valid = songs[split_index:]
    logger.info(f"Using {len(songs_train)} songs for training.")
    logger.info(f"Using {len(songs_valid)} songs for validation.")

    songs_data_train = preprocess_music21_songs(songs_train, train=True)
    songs_data_valid = preprocess_music21_songs(songs_valid, train=False)

    return songs_data_train, songs_data_valid


def preprocess_music21_songs(songs, train):
    #print("SONGS")

    songs_data = []
    for song in songs:
        song_data = preprocess_music21_song(song, train)
        if song_data is not None:
            songs_data += [song_data]

    return songs_data


def preprocess_music21_song(song, train):
    #print("  SONG", song.metadata.title, song.metadata.number)

    # Skip everything that has multiple measures and/or are not 4/4.
    meters = [meter.ratioString for meter in song.recurse().getElementsByClass(music21.meter.TimeSignature)]
    meters = list(set(meters))
    if len(meters) != 1:
        logger.debug(f"Skipping because of multiple measures.")
        return None
    elif meters[0] != "4/4":
        logger.debug(f"Skipping because of meter {meters[0]}.")
        return None

    song_data = {}
    song_data["title"] = song.metadata.title
    song_data["number"] = song.metadata.number
    song_data["tracks"] = []
    for part_index, part in enumerate(song.parts):
        track_data = preprocess_music21_part(part, part_index, train)
        song_data["tracks"] += [track_data]

    return song_data


def preprocess_music21_part(part, part_index, train):
    #print("    PART", part.partName)

    track_data = {}
    track_data["name"] = part.partName
    track_data["number"] = part_index
    track_data["bars"] = []

    for measure_index in range(1000):
        measure = part.measure(measure_index)
        if measure is None:
            break

        bar_data = preprocess_music21_measure(measure, train)
        track_data["bars"] += [bar_data]
    return track_data


def preprocess_music21_measure(measure, train):
    #print("      MEASURE")

    bar_data = {}
    bar_data["events"] = []

    events = []
    for note in measure.recurse(classFilter=("Note")):
        #print("        NOTE", note.pitch.midi, note.offset, note.duration.quarterLength)
        events += [("NOTE_ON", note.pitch.midi, 4 * note.offset)]
        events += [("NOTE_OFF", note.pitch.midi, 4 * note.offset + 4 * note.duration.quarterLength)]

    bar_data["events"] = events_to_events_data(events)
    return bar_data

    events = sorted(events, key=lambda event: event[2])
    for event_index, event, event_next in zip(range(len(events)), events, events[1:] + [None]):
        if event_index == 0 and event[2] != 0.0:
            event_data = {
                "type": "TIME_DELTA",
                "delta": event[2]
            }
            bar_data["events"] += [event_data]

        event_data = {
            "type": event[0],
            "pitch": event[1]
        }
        bar_data["events"] += [event_data]

        if event_next is None:
            continue

        delta = event_next[2] - event[2]
        assert delta >= 0, events
        if delta != 0.0:
            event_data = {
                "type": "TIME_DELTA",
                "delta": delta
            }
            bar_data["events"] += [event_data]

    return bar_data
