import argparse
import os
import numpy as np
import pickle
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.datasets import conf
from typing import List, Dict
from torch.utils.data import DataLoader
from torchsig.utils.dataset import SignalDataset
from torchsig.utils.visualize import IQVisualizer
import matplotlib
import matplotlib.pyplot as plt
import random

class SignalGenerator:
    def __init__(self, num_samples: int, classes: List[str], num_iq_samples: int = 1_000_000, use_class_idx: bool = True, save_path: str = "./", file_name: str = "generated_signals.pkl"):
        self.num_samples = num_samples
        self.classes = classes
        self.impaired = False
        self.level = 0
        self.num_iq_samples = num_iq_samples
        self.use_class_idx = use_class_idx
        self.save_path = save_path
        self.file_name = file_name

        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.config = self._select_config()
        self.dataset = self._create_dataset()
        self.signals = []

    def _select_config(self):
        config_class = conf.Sig53CleanTrainConfig
        return config_class(
            name="{}_signal_generation".format("clean"),
            num_samples=self.num_samples,
            level=self.level,
            num_iq_samples=self.num_iq_samples,
            use_class_idx=self.use_class_idx,
        )

    def _create_dataset(self):
        return ModulationsDataset(
            level=self.level,
            classes=self.classes,
            num_samples=self.num_samples,
            num_iq_samples=self.num_iq_samples,
            use_class_idx=self.use_class_idx,
            include_snr=False,
            eb_no=self.config.eb_no,
        )

    def generate(self):
        for idx in range(self.num_samples):
            sample, label = self.dataset[idx]
            sample = sample.astype(np.complex64)
            signal_info = {
                "sample": sample,
                "label_index": label,
                "label_class": self.idx_to_class[label],
            }
            mod_class = self.idx_to_class[label]

            if "ofdm" in mod_class:
                additional_info = (
                    "num_subcarriers: [64, 128, 256, 512, 1024, 2048], "
                    "cyclic_prefix_ratios: (0.125, 0.25), "
                    "sidelobe_suppression_methods: ('none', 'lpf', 'rand_lpf', 'win_start', 'win_center'), "
                    "dc_subcarrier: ('on', 'off'), "
                    "time_varying_realism: ('off',), "
                    "constellations: ['bpsk', 'qpsk', '16qam', '64qam', '256qam', '1024qam']"
                )
            else:
                additional_info = "iq_samples_per_symbol: 2"

            signal_info["additional_info"] = additional_info
            self.signals.append(signal_info)
        print(f"Generated {len(self.signals)} signals")

    def save_iq_file(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        file_path = os.path.join(self.save_path, self.file_name)
        with open(file_path, "wb") as f:
            pickle.dump(self.signals, f)

    def retrieve_signal(self, idx):
        signal_info = self.signals[idx]
        sample, label = self.dataset[idx]
        sample = sample.astype(np.complex64)
        return {
            "sample": sample,
            "class_index": label,
            "class_name": self.idx_to_class[label],
            "signal_info": signal_info
        }

    def save_plot(self):
        class DataWrapper(SignalDataset):
            def __init__(self, dataset):
                self.dataset = dataset
                super().__init__(dataset)

            def __getitem__(self, idx):
                sample = self.dataset[idx]
                return sample

            def __len__(self) -> int:
                return len(self.dataset)

        plot_dataset = DataWrapper(self.dataset)
        data_loader = DataLoader(dataset=plot_dataset, batch_size=16, shuffle=False)

        def target_idx_to_name(tensor: np.ndarray) -> List[str]:
            batch_size = tensor.shape[0]
            labels = []
            for idx in range(batch_size):
                labels.append(self.idx_to_class[int(tensor[idx])])
            return labels

        visualizer = IQVisualizer(
            data_loader=data_loader,
            visualize_transform=None,
            visualize_target_transform=target_idx_to_name,
        )

        for figure in iter(visualizer):
            figure.set_size_inches(14, 9)
            plot_path = os.path.join(self.save_path, 'signals_plot.png')
            figure.savefig(plot_path)
            print(f"Plot saved to {plot_path}")
            break

def main():
    parser = argparse.ArgumentParser(description="Signal Generator")
    parser.add_argument('--classes', type=str, required=True, help='list of included classes')

    parser.add_argument('--num_iq_samples', type=int, default=1_000_000, help='Number of IQ samples per Signal')

    parser.add_argument('--save_path', type=str, default="./", help='Path to save the generated signals')

    parser.add_argument('--file_name', type=str, default="generated_signals.pkl", help='File name for the generated signals')

    parser.add_argument('--save_plot', action='store_true', help='Flag to save the visualization as a file')

    args = parser.parse_args()

    classes = args.classes.split(',')
    num_samples = len(classes)

    signal_generator = SignalGenerator(
        num_samples=num_samples,
        classes=classes,
        num_iq_samples=args.num_iq_samples,
        save_path=args.save_path,
        file_name=args.file_name,
    )

    signal_generator.generate()
    signal_generator.save_iq_file()

    random_number = random.randrange(num_samples)

    sample_signal = signal_generator.retrieve_signal(random_number)
    print("Sample Signal Info:", {k: sample_signal[k] for k in sample_signal if k != 'sample'})

    if args.save_plot:
        signal_generator.save_plot()

if __name__ == "__main__":
    main()
