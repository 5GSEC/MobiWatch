#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.
# SPDX-FileCopyrightText: 2019-present Open Networking Foundation <info@opennetworking.org>
#
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup


setup(
    name="deep-watch",
    packages=find_packages(),
    python_requires=">=3.7",
    #install_requires=["onos-ric-sdk-python>=0.2.5"],
)

