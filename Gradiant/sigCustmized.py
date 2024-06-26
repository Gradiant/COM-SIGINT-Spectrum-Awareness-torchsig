import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import lmdb
import numpy as np

from torchsig.datasets import conf
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.transforms import Identity
from torchsig.utils.types import SignalData, SignalDescription


class Sig53:

    selected_classes = [
        "bpsk",
        "qpsk",
        "8psk",
        "16qam",
        "64qam",
        "256qam",
        "ofdm-64",
        "ofdm-72",
        "ofdm-128",
        "ofdm-180",
        "ofdm-256",
        "ofdm-300",
        "ofdm-512",
        "ofdm-600",
        "ofdm-900",
        "ofdm-1024",
        "ofdm-1200",
        "ofdm-2048",
    ]

    _idx_to_name_dict = dict(zip(range(len(selected_classes)),selected_classes))
    _name_to_idx_dict = dict(zip(selected_classes, range(len(selected_classes))))

    @staticmethod
    def convert_idx_to_name(idx: int) -> str:
        return Sig53._idx_to_name_dict.get(idx, "unknown")

    @staticmethod
    def convert_name_to_idx(name: str) -> int:
        return Sig53._name_to_idx_dict.get(name, -1)

    def __init__(
        self,
        root: str,
        train: bool = True,
        impaired: bool = True,
        eb_no: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        use_signal_data: bool = False,
    ):
        self.root = Path(root)
        self.train = train
        self.impaired = impaired
        self.eb_no = eb_no
        self.use_signal_data = use_signal_data

        self.T = transform if transform else Identity()
        self.TT = target_transform if target_transform else Identity()

        cfg: conf.Sig53Config = (
            "Sig53"  # type: ignore
            + ("Impaired" if impaired else "Clean")
            + ("EbNo" if (impaired and eb_no) else "")
            + ("Train" if train else "Val")
            + "Config"
        )

        cfg = getattr(conf, cfg)()  # type: ignore

        self.path = self.root / cfg.name
        self.env = lmdb.Environment(
            str(self.path).encode(), map_size=int(1e12), max_dbs=2, lock=False
        )
        self.data_db = self.env.open_db(b"data")
        self.label_db = self.env.open_db(b"label")
        with self.env.begin(db=self.data_db) as data_txn:
            self.length = data_txn.stat()["entries"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        encoded_idx = pickle.dumps(idx)
        with self.env.begin(db=self.data_db) as data_txn:
            iq_data = pickle.loads(data_txn.get(encoded_idx)).numpy()

        with self.env.begin(db=self.label_db) as label_txn:
            mod, snr = pickle.loads(label_txn.get(encoded_idx))

        mod = int(mod.numpy())
        if self.use_signal_data:
            signal_desc = SignalDescription(
                class_name=self._idx_to_name_dict[mod],
                class_index=mod,
                snr=snr,
            )
            data: SignalData = SignalData(
                data=deepcopy(iq_data.tobytes()),
                item_type=np.dtype(np.float64),
                data_type=np.dtype(np.complex128),
                signal_description=[signal_desc],
            )
            data = self.T(data)  # type: ignore
            target = self.TT(data.signal_description)  # type: ignore
            assert data.iq_data is not None
            sig_iq_data: np.ndarray = data.iq_data
            return sig_iq_data, target

        np_data: np.ndarray = self.T(iq_data)  # type: ignore
        target = (self.TT(mod), snr)  # type: ignore

        return np_data, target
