from typing import Callable, Tuple, Dict, Any
import torch


SelfDistilLosses = Tuple[Dict[str, torch.Tensor], torch.Tensor]
SelfDistilLossesCalculator = Callable[[Any, Any], SelfDistilLosses]