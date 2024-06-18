"""
test_sionna.py

Project: None

Maintainer Anxo Tato Arias (you@you.you)
Created @ Monday, 11th December 2023 9:16:14 am

Copyright (c) 2023 Gradiant
All Rights Reserved
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from dl_sigint.data_science.sionna_class import (
    Channel_Type,
    get_random_channel_type,
    Sionna_Channel,
)

from dl_sigint.data.data_utils import (
    convert_from_torch_to_numpy,
    convert_from_numpy_to_torch,
)


def test_channel_type():
    # Test 1
    # Create an instance of Channel_Type
    channel_1 = Channel_Type("TDL", "A", 100, 3.975e9, 0, 3, "UMi")
    channel_2 = Channel_Type("TDL", "B", 100, 3.975e9, 1, 3, "UMa")
    channel_3 = Channel_Type("TDL", "D", 100, 3.975e9, 2, 3, "RMa")
    channel_4 = Channel_Type("TDL", "E", 100, 3.975e9, 0, 3, "Indoor")


def test_random_Sionna_channel_type():
    # Test 2
    # Get a random Channel_Type object
    channel_type = get_random_channel_type()

    # Check if channel_type is an instance of Channel_Type
    assert isinstance(channel_type, Channel_Type)

    # Check the model, type and delay_spread_ns
    assert channel_type.model == "TDL"
    assert channel_type.type in ["A", "B", "C", "D", "E"]
    assert channel_type.delay_spread >= 0 and channel_type.delay_spread <= 1148


def test_create_sionna_channel():
    # Test 3
    # Get a random Channel_Type object
    random_channel = get_random_channel_type()
    assert isinstance(random_channel, Channel_Type)

    # Create an instance of Sionna_Channel
    channel = Sionna_Channel(random_channel, 3.975e9, 61.44e6, 8192)
    assert isinstance(channel, Sionna_Channel)


def test_apply_channel():
    # Test 4
    batch_size = 10
    num_time_samples = 8192
    signal = np.random.randn(batch_size, num_time_samples) + 1j * np.random.randn(
        batch_size, num_time_samples
    )

    # Get a random Channel_Type object
    channel = get_random_channel_type()

    # Create an instance of Sionna_Channel
    channel = Sionna_Channel(channel, 3.975e9, 61.44e6, 8192)

    output = channel.apply_channel(signal)

    assert isinstance(output, np.ndarray)


def test_convert_and_check():
    # Test 5
    batch_size = 10
    num_time_samples = 8192
    signal = np.random.randn(batch_size, num_time_samples) + 1j * np.random.randn(
        batch_size, num_time_samples
    )

    assert signal.shape == (batch_size, num_time_samples)

    signal_torch = convert_from_numpy_to_torch(signal, batch_size, num_time_samples)
    assert signal_torch.shape == torch.Size([batch_size, 1, 2, num_time_samples])

    signal_numpy = convert_from_torch_to_numpy(signal_torch)
    assert signal_numpy.shape == (batch_size, num_time_samples)

    np.allclose(signal, signal_numpy, rtol=1e-05, atol=1e-08, equal_nan=False)

    signal_torch_2 = convert_from_numpy_to_torch(signal_numpy, batch_size, num_time_samples)
    assert signal_torch_2.shape == torch.Size([batch_size, 1, 2, num_time_samples])

    np.allclose(signal_torch, signal_torch_2, rtol=1e-05, atol=1e-08, equal_nan=False)


def test_convert_and_check_2():
    # Test 6
    batch_size = 5
    num_time_samples = 8192
    signal = np.random.randn(batch_size, num_time_samples) + 1j * np.random.randn(
        batch_size, num_time_samples
    )

    assert signal.shape == (batch_size, num_time_samples)

    signal_torch = convert_from_numpy_to_torch(signal, batch_size, num_time_samples)
    assert signal_torch.shape == torch.Size([batch_size, 1, 2, num_time_samples])

    signal_numpy = convert_from_torch_to_numpy(signal_torch)
    assert signal_numpy.shape == (batch_size, num_time_samples)

    np.allclose(signal, signal_numpy, rtol=1e-05, atol=1e-08, equal_nan=False)

    signal_torch_2 = convert_from_numpy_to_torch(signal_numpy, batch_size, num_time_samples)
    assert signal_torch_2.shape == torch.Size([batch_size, 1, 2, num_time_samples])

    np.allclose(signal_torch, signal_torch_2, rtol=1e-05, atol=1e-08, equal_nan=False)



# ToDo Ivana:

def test_Sionna_Data_Augmentation():

    #prob AWGN=1.0 (output_batch_input_batch

    # output_batch.size = input_batch_size....