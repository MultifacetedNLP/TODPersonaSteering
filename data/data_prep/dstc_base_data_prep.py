import itertools
import logging
from typing import Dict
from itertools import chain

logger = logging.getLogger(__name__)

import hydra
from omegaconf import DictConfig
from config.python.dataprep_config import DataPrepConfig
from pathlib import Path
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from data.data_prep.data_prep_strategy_resolver import DataPrepStrategyResolver
from my_enums import Steps
from utils import get_csv_data_path, get_dialog_file_paths
from data.tod.turns.turn_csv_row_base import TurnCsvRowBase
from data.tod.turns.turn_csv_row_factory import TurnCsvRowFactory
import utils
import numpy as np
from data.sgd_dstc8.dstc_dataclasses import get_schemas, DstcSchema, DstcDialog
from data.data_prep.data_prep_strategy import DataPrepStrategy


class DstcBaseDataPrep:
    def __init__(self, cfg: DataPrepConfig, data_prep_strategy: DataPrepStrategy):
        self.cfg = cfg
        self.data_prep_strategy = data_prep_strategy

    def _prepare_dialog_file(
        self,
        path: Path,
        schemas: Dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> np.ndarray:
        data = []
        dialog_json_data = utils.read_json(path)
        for d in dialog_json_data:
            dialog = DstcDialog.from_dict(d)
            prepped_dialog = self.data_prep_strategy.prepare_dialog(
                dialog, schemas, turn_csv_row_handler
            )
            if prepped_dialog is None:
                continue
            data.append(prepped_dialog)
        if not len(data):
            return np.array(data)
        return list(chain.from_iterable(data)) # np.concatenate(data, axis=0)

    def run(self):
        steps = Steps.list()
        schemas = {}
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in steps]:
            schemas.update(d)
        turn_csv_row_handler: TurnCsvRowBase = TurnCsvRowFactory.get_handler(self.cfg)
        step_dir = Path(self.cfg.processed_data_root / self.cfg.step_name)
        step_dir.mkdir(parents=True, exist_ok=True)
        dialog_paths = get_dialog_file_paths(self.cfg.raw_data_root, self.cfg.step_name)
        # schemas = self._get_schemas(step)
        out_data = []
        if self.cfg.num_dialogs == "None":
            self.cfg.num_dialogs = len(dialog_paths)
        csv_file_path = get_csv_data_path(
            step=self.cfg.step_name,
            num_dialogs=self.cfg.num_dialogs,
            cfg=self.cfg,
        )
        if csv_file_path.exists() and not self.cfg.overwrite:
            logger.info("%s csv file already exists and overwrite is false, so skipping", self.cfg.step_name)
            return

        if self.cfg.data_prep_multi_process:
            res = list(
                tqdm(
                    Pool().imap(
                        self._prepare_dialog_file,
                        dialog_paths[: self.cfg.num_dialogs],
                        itertools.repeat(schemas),
                        itertools.repeat(turn_csv_row_handler),
                    ),
                    total=self.cfg.num_dialogs,
                )
            )
        else:
            res = []
            for d in tqdm(dialog_paths[: self.cfg.num_dialogs]):
                output = self._prepare_dialog_file(d, schemas, turn_csv_row_handler)
                if output is not None:
                    res.append(output)

        out_data = [d for d in res if len(d)]
        headers = turn_csv_row_handler.get_csv_headers(self.cfg.should_add_schema)
        if len(out_data) == 0:
            domains = ",".join(self.cfg.domain_setting)
            logger.warning("No data for %s: %s", self.cfg.step_name, domains)
            return
        csv_data = list(chain.from_iterable(out_data)) # np.concatenate(out_data, axis=0)
        utils.write_csv(headers, csv_data, csv_file_path)


@hydra.main(config_path="../../config/data_prep/", config_name="dstc_base_data_prep", version_base="1.1")
def hydra_start(cfg: DictConfig) -> None:
    dpconf = DataPrepConfig(**cfg)
    dp_strategy = DataPrepStrategyResolver.resolve(dpconf)
    stdp = DstcBaseDataPrep(dpconf, dp_strategy)
    stdp.run()


if __name__ == "__main__":
    hydra_start()
