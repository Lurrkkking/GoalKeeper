# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .g1.g1_29_config import G129Cfg, G129CfgPPO
from .g1.g1_extreme_dive_config import G1ExtremeDiveCfg, G1ExtremeDiveCfgPPO
from .q1.q1_goalkeeper_config import (
    Q1GoalkeeperCfg, Q1GoalkeeperCfgPPO, Q1GoalkeeperCfgHard,
    Q1GoalkeeperCfgMediumDR, Q1GoalkeeperCfgContactDR, Q1GoalkeeperCfgFinetuneDR,
    Q1GoalkeeperCfgStage1,
)
from .q1.q1_extreme_dive_config import Q1ExtremeDiveCfg, Q1ExtremeDiveCfgPPO
from .g1.g1_dive_reach_config import G1DiveReachCfg, G1DiveReachCfgPPO, DiveReachRobot

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "29", LeggedRobot, G129Cfg(), G129CfgPPO() )
task_registry.register( "q1", LeggedRobot, Q1GoalkeeperCfg(), Q1GoalkeeperCfgPPO() )
task_registry.register( "q1_hard", LeggedRobot, Q1GoalkeeperCfgHard(), Q1GoalkeeperCfgPPO() )
task_registry.register( "q1_medium", LeggedRobot, Q1GoalkeeperCfgMediumDR(), Q1GoalkeeperCfgPPO() )
task_registry.register( "q1_contact", LeggedRobot, Q1GoalkeeperCfgContactDR(), Q1GoalkeeperCfgPPO() )
task_registry.register( "q1_finetune", LeggedRobot, Q1GoalkeeperCfgFinetuneDR(), Q1GoalkeeperCfgPPO() )
task_registry.register( "q1_stage1", LeggedRobot, Q1GoalkeeperCfgStage1(), Q1GoalkeeperCfgPPO() )
task_registry.register( "q1_extreme_dive", LeggedRobot, Q1ExtremeDiveCfg(), Q1ExtremeDiveCfgPPO() )
task_registry.register( "g1_extreme_dive", LeggedRobot, G1ExtremeDiveCfg(), G1ExtremeDiveCfgPPO() )
task_registry.register( "g1_dive_reach", DiveReachRobot, G1DiveReachCfg(), G1DiveReachCfgPPO() )
