import numpy as np

# All trainers.
trainers = {}


class Trainer:
    """
    Trainer class for training model on given (partially) datasets.
    """

    def __init__(self, trainer_config, dataset, model_fn, model_config, metric, get_feature_fn):
        """
        :param Dict trainer_config: Dictionary of hyper-parameters of the trainer.
        :param ALDataset dataset: An initial ALDataset.
        :param model_fn: Function for instantiating a model.
        :param Dict model_config: Dictionary of hyper-parameters for instantiating a model.
        :param Metric metric: Metric object for tracking performances.
        """
        self.trainer_config = trainer_config
        self.dataset = dataset
        self.model_fn = model_fn
        self.model_config = model_config
        self.metric = metric
        self._eval_results = [None for _ in range(12)]
        self.get_feature_fn = get_feature_fn

    # TODO: check if this is implemented properly for semiSL methods
    def __init_subclass__(cls, **kwargs):
        """Register trainer subclasses."""
        super().__init_subclass__(**kwargs)
        trainers[cls.trainer_name] = cls

    def train(self, finetune_model=None, finetune_config=None):
        """
        Train a model.

        :param bool log: Flag indicating whether to log training metrics.
        :param Optional[torch.nn.Module] finetune_model: Warm start model if indicated.
        :param Optional[Dict] finetune_config: Warm start model hyper-parameters.
        """
        raise NotImplementedError("Subclass does not have implementation of training function.")

    def evaluate_on_train(self, model, mc_dropout=False):
        self._eval_results[0], self._eval_results[3], self._eval_results[6], self._eval_results[9] = \
            self._test("train", model, mc_dropout=mc_dropout)

    def evaluate_on_val(self, model, mc_dropout=False):
        self._eval_results[1], self._eval_results[4], self._eval_results[7], self._eval_results[10] = \
            self._test("val", model, mc_dropout=mc_dropout)

    def evaluate_on_test(self, model, mc_dropout=False):
        self._eval_results[2], self._eval_results[5], self._eval_results[8], self._eval_results[11] = \
            self._test("test", model, mc_dropout=mc_dropout)

    def retrieve_inputs(self, input_types):
        inputs = [np.array(self._eval_results[int(t)]) for t in input_types]
        return inputs

    def compute_metric(self, epoch):
        metric_dict = self.metric.compute(epoch, self._eval_results[0], self._eval_results[3], self._eval_results[6],
                                          self._eval_results[1], self._eval_results[4], self._eval_results[7],
                                          self._eval_results[2], self._eval_results[5], self._eval_results[8],
                                          num_labeled=self.dataset.num_labeled(),
                                          labeled=self.dataset.labeled_idxs())
        return metric_dict

    def _test(self, dataset, model, **kwargs):
        raise NotImplementedError("Subclass does not have implementation of testing function.")
