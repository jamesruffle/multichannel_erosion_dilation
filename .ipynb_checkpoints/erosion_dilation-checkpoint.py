# Copyright (c) James Ruffle
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dictionary-based transforms for multi-channel erosion and dilation of marks, for example in the erosion or dilation of synthetic lesion
"""

from typing import Any, Dict, Hashable, Mapping, Optional
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.config.type_definitions import NdarrayOrTensor
import scipy
import numpy as np

class BinaryErosiond(RandomizableTransform, MapTransform):
    """
    Dictionary-based version for Binary Erosion transform:py:class:
    Author: James Ruffle | j.ruffle@ucl.ac.uk | 25/1/24
    """
    def __init__(
        self,
        keys: KeysCollection,
        iteration: int = 1,
        random_iteration: bool = True,
        iteration_low: int = 1,
        iteration_high: int = 5,
        prob: float = 0.1,
        vary_across_channels: bool = False,
        allow_missing_keys: bool = True,
        verbose: bool = False,
        
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of augmentation application.
                (Default 0.1, with 10% probability that Binary Erosion will occur)
            iteration: Specify a specific number of iterations to be run.
                (Default 1)
            random_iteration: Specify to take a random number of iterations between the range iteration_low, iteration_high
                (Default True)
            iteration_low: lower limit for the random integer selection for number of iterations for Binary Erosion
                (Default 1)
            iteration_high: upper limit for the random integer selection for number of iterations for Binary Erosion
                (Default 5)
            vary_across_channels: whether to vary the random integer selection across input keys
                (Default False - i.e., number of iterations in Binary Erosion will be the same across all keys)
            allow_missing_keys: don't raise exception if key is missing.
        """
        
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self,prob=prob)
        
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.iteration = iteration
        self.random_iteration = random_iteration
        self.iteration_low = iteration_low
        self.iteration_high = iteration_high
        self.vary_across_channels = vary_across_channels
        self.verbose = verbose
        

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "BinaryErosiond":
        super().set_random_state(seed, state)
        return self
    
    def randomize(self: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        if self.random_iteration and self._do_transform:
            if self.verbose:
                print('Choosing random iteration within the low to high range')
            self.iterations = self.R.randint(low=self.iteration_low, high=self.iteration_high)
        if not self.random_iteration and self._do_transform:
            if self.verbose:
                print('Using fixed number of iterations')
            self.iterations = self.iteration

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        keys_to_process = list(set(list(d)).intersection(set(list(self.keys))))
        if len(keys_to_process) == 0:
            """
            This is only a problem if we do not allow missing keys
            """
            if not self.allow_missing_keys:
                if self.verbose:
                    print("No keys to process. Exiting.")
                quit()
            else:
                return d
        else:

            self.randomize()
            
            if not self._do_transform:
                if self.verbose:
                    print('Not doing transform')
                return d

            if self._do_transform:
                if self.verbose:
                    print('Doing transform')
                for key in self.keys:
                    if key not in d and not self.allow_missing_keys:
                        if self.verbose:
                            print("key not found. Exiting.")
                        quit()
                    if key in d:
                        if self.vary_across_channels and self.random_iteration:
                            self.randomize()
                            if self.verbose:
                                print('vary across channels')
                        if self.verbose:
                            print('Running with # iterations: '+str(self.iterations))
                        d[key].array = scipy.ndimage.binary_erosion(d[key],iterations=self.iterations).astype('float32')

            return d


class BinaryDilationd(RandomizableTransform, MapTransform):
    """
    Dictionary-based version for Binary Dilation transform:py:class:
    Author: James Ruffle | j.ruffle@ucl.ac.uk | 25/1/24
    """
    def __init__(
        self,
        keys: KeysCollection,
        iteration: int = 1,
        random_iteration: bool = True,
        iteration_low: int = 1,
        iteration_high: int = 5,
        prob: float = 0.1,
        vary_across_channels: bool = False,
        allow_missing_keys: bool = True,
        verbose: bool = False,
        
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of augmentation application.
                (Default 0.1, with 10% probability that Binary Dilation will occur)
            iteration: Specify a specific number of iterations to be run.
                (Default 1)
            random_iteration: Specify to take a random number of iterations between the range iteration_low, iteration_high
                (Default True)
            iteration_low: lower limit for the random integer selection for number of iterations for Binary Dilation
                (Default 1)
            iteration_high: upper limit for the random integer selection for number of iterations for Binary Dilation
                (Default 5)
            vary_across_channels: whether to vary the random integer selection across input keys
                (Default False - i.e., number of iterations in Binary Dilation will be the same across all keys)
            allow_missing_keys: don't raise exception if key is missing.
        """
        
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self,prob=prob)
        
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.iteration = iteration
        self.random_iteration = random_iteration
        self.iteration_low = iteration_low
        self.iteration_high = iteration_high
        self.vary_across_channels = vary_across_channels
        self.verbose = verbose
        

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "BinaryDilationd":
        super().set_random_state(seed, state)
        return self
    
    def randomize(self: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        if self.random_iteration and self._do_transform:
            if self.verbose:
                print('Choosing random iteration within the low to high range')
            self.iterations = self.R.randint(low=self.iteration_low, high=self.iteration_high)
        if not self.random_iteration and self._do_transform:
            if self.verbose:
                print('Using fixed number of iterations')
            self.iterations = self.iteration

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        keys_to_process = list(set(list(d)).intersection(set(list(self.keys))))
        if len(keys_to_process) == 0:
            """
            This is only a problem if we do not allow missing keys
            """
            if not self.allow_missing_keys:
                if self.verbose:
                    print("No keys to process. Exiting.")
                quit()
            else:
                return d
        else:

            self.randomize()
            
            if not self._do_transform:
                if self.verbose:
                    print('Not doing transform')
                return d

            if self._do_transform:
                if self.verbose:
                    print('Doing transform')
                for key in self.keys:
                    if key not in d and not self.allow_missing_keys:
                        if self.verbose:
                            print("key not found. Exiting.")
                        quit()
                    if key in d:
                        if self.vary_across_channels and self.random_iteration:
                            self.randomize()
                            if self.verbose:
                                print('vary across channels')
                        if self.verbose:
                            print('Running with # iterations: '+str(self.iterations))
                        d[key].array = scipy.ndimage.binary_dilation(d[key],iterations=self.iterations).astype('float32')

            return d