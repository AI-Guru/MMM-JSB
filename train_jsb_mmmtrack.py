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
 
import os
import sys
from source import datasetcreatorconfig
from source import datasetcreator
from source import mmmtrainerconfig
from source import mmmtrainer


# Create dataset if it does not exist yet.
dataset_creator_config = datasetcreatorconfig.JSBDatasetCreatorTrackConfig()
dataset_creator = datasetcreator.DatasetCreator(dataset_creator_config)
dataset_creator.create(datasets_path=os.path.join("datasets"), overwrite=False)

# Train the model.
trainer_config = mmmtrainerconfig.MMMTrainerBaseConfig(
    tokenizer_path = os.path.join("datasets", "jsb_mmmtrack", "tokenizer.json"),
    dataset_train_files=[os.path.join("datasets", "jsb_mmmtrack", "token_sequences_train.txt")],
    dataset_validate_files=[os.path.join("datasets", "jsb_mmmtrack", "token_sequences_valid.txt")],
    pad_length=768,
    shuffle_buffer_size=10000,
    batch_size=16,
    epochs=10,
)
trainer = mmmtrainer.MMMTrainer(trainer_config)
trainer.train(
    output_path=os.path.join("training/jsb_mmmtrack"),
    simulate="simulate" in sys.argv
    )
