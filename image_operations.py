from abc import ABC, abstractmethod
from typing import Optional, Dict, Type, List

import numpy as np


class ImageOperation(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        pass


class MosaicOperation(ImageOperation):
    def __init__(self, block_size: int = 10):
        self._block_size = block_size

    @property
    def name(self) -> str:
        return "Mosaic"

    @property
    def id(self) -> str:
        return "mosaic"

    @property
    def block_size(self) -> int:
        return self._block_size

    @block_size.setter
    def block_size(self, value: int):
        self._block_size = max(1, value)

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = image.copy()
        if not mask.any():
            return result

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return result

        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]

        for y in range(y0, y1 + 1, self._block_size):
            for x in range(x0, x1 + 1, self._block_size):
                block_slice = (
                    slice(y, min(y + self._block_size, image.shape[0])),
                    slice(x, min(x + self._block_size, image.shape[1]))
                )
                if mask[block_slice].any():
                    block = image[block_slice]
                    avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
                    result[block_slice] = avg_color
        return result


class OperationRegistry:
    _operations: Dict[str, Type[ImageOperation]] = {}

    @classmethod
    def register(cls, operation_class: Type[ImageOperation]):
        instance = operation_class()
        cls._operations[instance.id] = operation_class
        return operation_class

    @classmethod
    def get(cls, operation_id: str) -> Optional[ImageOperation]:
        op_class = cls._operations.get(operation_id)
        return op_class() if op_class else None

    @classmethod
    def list_operations(cls) -> List[str]:
        return list(cls._operations.keys())

    @classmethod
    def get_all(cls) -> List[ImageOperation]:
        return [op_class() for op_class in cls._operations.values()]


OperationRegistry.register(MosaicOperation)