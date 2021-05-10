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


def events_to_events_data(events):

    events = sorted(events, key=lambda event: event[2])

    events_data = []
    for event_index, event, event_next in zip(range(len(events)), events, events[1:] + [None]):
        if event_index == 0 and event[2] != 0.0:
            event_data = {
                "type": "TIME_DELTA",
                "delta": event[2]
            }
            events_data += [event_data]

        event_data = {
            "type": event[0],
            "pitch": event[1]
        }
        events_data += [event_data]

        if event_next is None:
            continue

        delta = event_next[2] - event[2]
        assert delta >= 0, events
        if delta != 0.0:
            event_data = {
                "type": "TIME_DELTA",
                "delta": delta
            }
            events_data += [event_data]

    return events_data
