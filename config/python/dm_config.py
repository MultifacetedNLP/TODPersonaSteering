from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union
from data.dstc.dstc_domains import DstcDomains
from my_enums import ContextType, MultiTaskNames, Steps
import data.dstc.dstc_utils as dstc_utils
from accelerate import Accelerator

if TYPE_CHECKING:
    from config.python.dm_config import DataModuleConfig
    from base_datamodule import StepData


class DataModuleConfig:
    def __init__(
        self,
        num_workers=8,
        batch_size=32,
        eval_batch_size=32,
        test_batch_size=32,
        data_split_percent: list[float] = None,
        project_root: str = None,
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        data_prep_out_root: str = "processed_data/simple_tod",
        max_token_len: int = 1024,
        test_prompt_max_len: int = 800,
        num_dialogs: list[int] = None,
        preprocessing_model_name="simple_tod",
        dataset_name="dstc",
        model_name="gpt2",
        tokenizer=None,
        delexicalize: bool = False,
        overwrite: list[bool] = None,
        num_turns: int = 26,
        train_domain_settings: Union[list[str], str] = None,
        dev_domain_settings: Union[list[str], str] = None,
        test_domain_settings: Union[list[str], str] = None,
        multi_domain = False,
        max_turns = 50,
        train_domain_percentage: float = 1.0,
        create_data_from_train: bool = False,
        create_data_from_train_splits: list[float] = None,
        is_scale_grad: bool = False,
        is_multi_task: bool = False,
        is_multi_head: bool = False,
        multi_tasks: list[MultiTaskNames] = None,
        should_add_schema: bool = False,
        should_add_sys_actions: bool = False,
        should_add_user_actions: bool = False,
        single_action_neg_samples: int = 5,
        contrast_with: str = None,
        contrastive_max_token_len: int = 512,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        should_add_service_calls: bool = False,
        should_add_dsts: bool = False,
        data_prep_multi_process: bool = True,
        test_num_turns_groups: list[Tuple[int, int]] = None,
        train_step_data: "StepData" = None,
        accelerator: "Accelerator" = None,
        service_results_num_items: int = 1,
        hard_api_calls = False,
        should_add_results_slots = False,
        should_add_user_req_slots = False,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.accelerator = Accelerator()
        self.hard_api_calls = hard_api_calls
        self.num_workers = num_workers
        self.model_name = model_name
        self.preprocessing_model_name = preprocessing_model_name
        self.project_root = Path(project_root)
        self.processed_data_root = self.project_root / data_prep_out_root
        self.raw_data_root = raw_data_root
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.data_split_percent = data_split_percent
        self.max_token_len = max_token_len
        self.test_prompt_max_len = test_prompt_max_len
        self.num_dialogs = num_dialogs
        self.dataset_name = dataset_name
        self.datasets: any = {}
        self.tokenizer = tokenizer or dstc_utils.get_tokenizer(model_name)
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False] * len(Steps)
        self.num_turns = num_turns
        self.is_scale_grad = is_scale_grad
        self.is_multi_task = is_multi_task
        self.is_multi_head = is_multi_head
        self.multi_tasks = (
            multi_tasks if self.is_multi_task and multi_tasks else MultiTaskNames.list()
        )
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.train_domain_percentage = train_domain_percentage
        self.single_action_neg_samples = (
            single_action_neg_samples if single_action_neg_samples else 5
        )
        if self.is_scale_grad and not self.should_add_schema:
            raise ValueError(
                "is_scale_grad is true but should_add_schema is false, which is not allowed"
            )
        self.contrast_with = contrast_with
        self.contrastive_max_token_len = contrastive_max_token_len
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results
        self.should_add_service_calls = should_add_service_calls
        self.should_add_user_req_slots = should_add_user_req_slots
        self.should_add_dsts = should_add_dsts
        self.data_prep_multi_process = data_prep_multi_process
        self.train_domain_settings = train_domain_settings
        self.dev_domain_settings = dev_domain_settings
        self.test_domain_settings = test_domain_settings
        self.multi_domain = multi_domain
        self.max_turns = max_turns
        self.create_data_from_train = create_data_from_train
        self.create_data_from_train_splits = create_data_from_train_splits or [0.1, 0.1]
        # these two variables are added so that we can have typing in DataPrepConfig.from_dm_config method
        self.step_name = None
        self.domain_setting = None
        self.test_num_turns_groups = test_num_turns_groups
        self.train_step_data = train_step_data
        self.service_results_num_items = service_results_num_items
        self.should_add_results_slots = should_add_results_slots
