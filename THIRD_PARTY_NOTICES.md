# Third-party notices

This file is the single list of third-party code, models and data used by this repository.

Part 1 lists files that are our own code (re-implementations or ports) following MIT-licensed upstream projects. They are distributed under the licence of this repository; the upstream copyright and permission notices are reproduced here as the MIT licence requires. No upstream source file is redistributed verbatim.

Part 2 lists third-party software, models and data that are used; their original files are not redistributed in this repository.

# Part 1: ports of MIT-licensed code

## DreamerV3

- Upstream: danijar/dreamerv3 (https://github.com/danijar/dreamerv3), commit e01491fad6434b2245a3b8ca201dd7faedcc458c
- Used in: `baselines/rssm.py`
- PyTorch re-implementation of the DreamerV3 world model (RSSM with a multilayer-perceptron core instead of the block recurrent core, discrete latents with uniform mixing, symlog inputs, RMSNorm, LaProp with adaptive gradient clipping), written following the upstream code.
- Licence: MIT

```
Copyright (c) 2024 Danijar Hafner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## TD-MPC2

- Upstream: nicklashansen/tdmpc2 (https://github.com/nicklashansen/tdmpc2), commit e9f5932
- Used in: `baselines/tdmpc2.py`, `baselines/tdmpc2_h40.py`
- The layers NormedLinear, SimNorm and the mlp builder and the weight initialisation are close ports of the upstream code; the forecasting model around them is ours.
- Licence: MIT

```
MIT License

Copyright (c) Nicklas Hansen (2023).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MBCD

- Upstream: LucasAlegre/mbcd (https://github.com/LucasAlegre/mbcd), commit fc824b3eca720f8198ff2034c61f197521c29d70
- Used in: `baselines/mbcd.py`
- PyTorch port of the model-based context detection method (probabilistic ensemble, CUSUM change detection, model library), adapted to offline training on logged data.
- Licence: MIT

```
MIT License

Copyright (c) 2021 Lucas Alegre

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PETS (handful-of-trials)

- Upstream: kchua/handful-of-trials (https://github.com/kchua/handful-of-trials), commit 77fd8802cc30b7683f0227c90527b5414c0df34c
- Used in: `control/planner.py`
- The cross-entropy-method planner follows the upstream CEM optimizer.
- Licence: MIT

```
MIT License

Copyright (c) 2019 Kurtland Chua

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ESCP

- Upstream: FanmingL/ESCP (https://github.com/FanmingL/ESCP), commit 518f13f8b002d142f670f52d9ef34778e2c2d59f
- Used in: `datasets/escphc/escp.py`
- Port of the upstream environment-change schedule and task sampling used to generate the ESCP-HC dataset.
- Licence: MIT

```
MIT License

Copyright (c) 2021 Fan-Ming Luo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

# Part 2: used; original files not redistributed

## MTS3

- Upstream: ALRhub/MTS3 (https://github.com/ALRhub/MTS3), commit ab971c0ec43df7664c4b727ca69d5b8dd0cab5da
- Used in: `baselines/mts3.py`, which runs five model modules of the MTS3 authors inside our pipeline.
- The upstream repository has no licence file, so its code is not included here. `baselines/third_party/mts3/fetch_and_patch.sh` downloads the pinned commit, checks the sha256 of the five files (`baselines/third_party/mts3/SHA256SUMS`) and applies two edits with `baselines/third_party/mts3/apply_patch.py`: package-relative imports, and a mask of the actions that the manager's abstract-action aggregation may use.
- `baselines/third_party/mts3/apply_patch.py` contains a few single lines of the upstream file as anchors for the two edits.

## MuJoCo Menagerie

- Upstream: google-deepmind/mujoco_menagerie (https://github.com/google-deepmind/mujoco_menagerie), commit 6da8b971844845de6ab27315aacc3a3fea4ddb8e
- Used for: the Franka Panda (`franka_emika_panda`, Apache License 2.0) and UR5e (`universal_robots_ur5e`, BSD-3-Clause, Copyright 2018 ROS Industrial Consortium) robot models of the Panda and UR5e datasets, the stored frames in `outputs/examples/frames/` and the Fig. 4 image of the project page.
- Not redistributed: `datasets/assets/fetch_menagerie.sh` downloads the two model folders with their licence files, copies the repository licence next to them and checks every file against the sha256 lists `datasets/assets/menagerie_panda.sha256` and `datasets/assets/menagerie_ur5e.sha256`.
- The stored frames and the Fig. 4 image show renders of these models. The notice of the UR5e model (the file LICENSE in the folder `universal_robots_ur5e` of the pinned commit) is reproduced here:

```
Copyright 2018 ROS Industrial Consortium

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Gymnasium

- Gymnasium 1.3.0 (MIT License), installed with pip.
- The HalfCheetah and Hopper model files are read from the installed package (the Hopper file is checked by sha256); no Gymnasium file is redistributed.

## MuJoCo, PyTorch, NumPy, matplotlib, Pillow

Installed with pip (`environment/`), not redistributed: MuJoCo 3.11.0 (Apache License 2.0), PyTorch 2.5.1 and torchvision 0.20.1 (BSD-3-Clause), NumPy (BSD-3-Clause), matplotlib (matplotlib licence), Pillow (MIT-CMU licence).

## Compared models written by us

GrBAL and ReBAL (Nagabandi et al., ICLR 2019), MOLe (Nagabandi et al., ICLR 2019) and GPMM (Xu et al., 2020) are implementations written by us. They follow the original papers and, where noted in the files of `baselines/`, the behavior of the original public code; no upstream source file is included.

The control task follows GrBAL's joint-failure task and reward.

The torso-mass definition of ESCP-HC follows DORA (Zhang et al., ICML 2024).

## ALFA dataset

- Azarakhsh Keipour, Mohammadreza Mousaei and Sebastian Scherer. ALFA: A Dataset for UAV Fault and Anomaly Detection. The International Journal of Robotics Research 40(2-3), 2021. https://doi.org/10.1177/0278364920966642. Data: https://doi.org/10.1184/R1/12707963.v1, licence CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- Azarakhsh Keipour, Mohammadreza Mousaei and Sebastian Scherer. Automatic Real-time Anomaly Detection for Autonomous Aerial Vehicles. IEEE International Conference on Robotics and Automation (ICRA), 2019. https://doi.org/10.1109/ICRA.2019.8794286. Part of the ALFA recordings was published with this paper, and the ALFA authors ask users of those recordings to cite it.
- `data/alfa/` contains the ALFA sequences derived from these recordings; attribution and the statement of changes are in `DATA_LICENSE.md` and `datasets/cards/alfa.md`. The original files are downloaded by `datasets/alfa/download_alfa.sh`. No endorsement by the ALFA authors is implied.

## Citations of the simulation software

- MuJoCo: Emanuel Todorov, Tom Erez and Yuval Tassa. MuJoCo: A Physics Engine for Model-Based Control. IROS, 2012.
- Gymnasium: Mark Towers et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. NeurIPS Datasets and Benchmarks Track, 2025.
- MuJoCo Menagerie: Kevin Zakka, Yuval Tassa and the MuJoCo Menagerie Contributors. MuJoCo Menagerie: A Collection of High-Quality Simulation Models for MuJoCo. https://github.com/google-deepmind/mujoco_menagerie, 2022.

## Names and images

Franka Emika Panda and UR5e are names of their owners; this repository has no affiliation with them and is not endorsed by them.

The images in `docs/assets/` are figures of the paper, shown for reference; they are not covered by the MIT or CC BY 4.0 grants of this repository.
