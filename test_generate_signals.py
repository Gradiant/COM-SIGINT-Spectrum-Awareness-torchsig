from generate_signals import SignalGenerator
import pytest
import os
import numpy as np
import pickle
import random

default_classes = [
    "ook", "bpsk", "4pam", "4ask", "qpsk", "8pam", "8ask", "8psk", "16qam",
    "16pam", "16ask", "16psk", "32qam", "32qam_cross", "32pam", "32ask",
    "32psk", "64qam", "64pam", "64ask", "64psk", "128qam_cross", "256qam",
    "512qam_cross", "1024qam", "2fsk", "2gfsk", "2msk", "2gmsk", "4fsk",
    "4gfsk", "4msk", "4gmsk", "8fsk", "8gfsk", "8msk", "8gmsk", "16fsk",
    "16gfsk", "16msk", "16gmsk", "ofdm-64", "ofdm-72", "ofdm-128", "ofdm-180",
    "ofdm-256", "ofdm-300", "ofdm-512", "ofdm-600", "ofdm-900", "ofdm-1024",
    "ofdm-1200", "ofdm-2048"
]


@pytest.fixture
def signal_generator_instance():
    # Choose a random subset of classes for testing
    test_classes = random.sample(default_classes, 5)
    num_samples = len(test_classes)
    save_path = "./test_output"
    file_name = "test_generated_signals.pkl"

    signal_generator = SignalGenerator(
        num_samples=num_samples,
        classes=test_classes,
        save_path=save_path,
        file_name=file_name,
    )
    signal_generator.generate()
    signal_generator.save_iq_file()

    yield signal_generator

    # Cleanup
    if os.path.exists(save_path):
        os.remove(os.path.join(save_path, file_name))
        os.rmdir(save_path)


# Check if number of generated signals equals to number of classes
def test_generate_signals(signal_generator_instance):
    assert len(signal_generator_instance.signals) == signal_generator_instance.num_samples

    # Check if all generated signals are unique
    signals_data = [signal["data"] for signal in signal_generator_instance.signals]
    # Convert each signal data array to a tuple for hashing and compare lengths
    unique_signals_data = set(map(lambda x: tuple(x), signals_data))
    assert len(unique_signals_data) == len(signals_data), "Generated signals are not unique"


def test_arguments(signal_generator_instance):
    # Check if the plot file is saved correctly
    signal_generator_instance.save_plot()
    plot_path = os.path.join(signal_generator_instance.save_path, 'signals_plot.png')
    assert os.path.exists(plot_path)
    os.remove(plot_path)

    # Check if the save_path and file_name arguments work correctly
    file_path = os.path.join(signal_generator_instance.save_path, signal_generator_instance.file_name)
    assert os.path.exists(file_path)

    # Check if num_iq_samples argument is handled correctly
    assert signal_generator_instance.num_iq_samples == 1_000_000

def test_saved_file_format(signal_generator_instance):
    file_path = os.path.join(signal_generator_instance.save_path, signal_generator_instance.file_name)
    assert os.path.exists(file_path)
    with open(file_path, "rb") as f:
        signals = pickle.load(f)
    assert isinstance(signals, list)
    for signal in signals:
        assert "data" in signal
        assert "label_index" in signal
        assert "label_class" in signal
        assert "additional_info" in signal

        # Check data types
        assert signal["data"].dtype == np.complex64
        assert isinstance(signal["label_index"], int)
        assert isinstance(signal["label_class"], str)
        assert isinstance(signal["additional_info"], str)

def test_number_of_signals(signal_generator_instance):
    assert len(signal_generator_instance.signals) == len(signal_generator_instance.classes)

if __name__ == "__main__":
    pytest.main([__file__])
