# Copyright 2021 Google LLC
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
# ==============================================================================
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3
WORKDIR /root

WORKDIR /

# Install pip reqs from both user and default
# NOTE: for this implementation, requirements.txt specifies 
#   the tornado, scikit-learn, and joblib libraries in 
#   the format: [library]==[version]. Build the requirements.txt
#   file to match your needs
#RUN pip install -r requirements.txt

RUN mkdir prediction

# Copies the trainer code to the docker image.
COPY sklearn_deploy/prediction/predict/ /prediction

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "prediction.predict"]