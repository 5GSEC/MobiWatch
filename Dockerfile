# SPDX-FileCopyrightText: Copyright 2004-present Facebook. All Rights Reserved.
# SPDX-FileCopyrightText: 2019-present Open Networking Foundation <info@opennetworking.org>
#
# SPDX-License-Identifier: Apache-2.0

FROM python:3.8-slim

# install all deps
WORKDIR /usr/local

# prepare gprpc dependencies
RUN pip install grpcio grpcio-tools

# prepare AI ML depdencies
RUN pip install numpy torch==2.0.1+cpu scikit-learn==1.3.2 -f https://download.pytorch.org/whl/torch_stable.html

# COPY onos_e2_sm ./onos_e2_sm
# RUN pip install --upgrade pip ./onos_e2_sm --no-cache-dir

# speed up subsequent image builds by pre-dl and pre-installing pre-reqs
COPY deepwatch/setup.py ./deepwatch/setup.py
RUN pip install ./deepwatch --no-cache-dir

# install actual app code
COPY deepwatch ./deepwatch
RUN pip install ./deepwatch --no-cache-dir

ENTRYPOINT [ "python" ]
