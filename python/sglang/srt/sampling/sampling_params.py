# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sampling parameters for text generation."""

# ============================================================================
# 文件功能说明 (File Purpose)
# ============================================================================
# 本文件定义了文本生成的所有采样参数 (sampling parameters)
# This file defines all user-controllable sampling parameters for text generation
#
# 【比喻理解】这个文件就像是"文本生成的控制面板"
# [Metaphor] This file is like the "control panel" for text generation
# - temperature: 调节创造性的旋钮 (knob for creativity)
# - top_p/top_k: 候选词的筛选器 (filters for candidate tokens)
# - max_new_tokens: 生成长度的限制器 (limiter for generation length)
# - stop conditions: 生成的刹车系统 (braking system for generation)
#
# 【核心参数概览 Core Parameters Overview】
# 1. 长度控制 (Length Control): max_new_tokens, min_new_tokens
# 2. 随机性控制 (Randomness Control): temperature, top_p, top_k, min_p
# 3. 重复抑制 (Repetition Suppression): frequency/presence/repetition_penalty
# 4. 停止条件 (Stop Conditions): stop, stop_token_ids, stop_regex
# 5. 结构化输出 (Structured Output): json_schema, regex, ebnf
# ============================================================================

import logging
from typing import Any, Dict, List, Optional, Union

# sre_parse is deprecated in Python 3.11+, use re._parser instead
try:
    import re._parser as sre_parse
except ImportError:
    import sre_parse  # Python < 3.11

_SAMPLING_EPS = 1e-6
TOP_K_ALL = 1 << 30

logger = logging.getLogger(__name__)


# ============================================================================
# SamplingParams: 文本生成的中央配置类
# SamplingParams: The Central Configuration Class for Text Generation
# ============================================================================
# 这个类是用户控制文本生成行为的唯一接口
# This class is the single interface for users to control text generation behavior
#
# 【设计理念】一次创建，到处使用 (Create once, use everywhere)
# 用户创建一个 SamplingParams 对象，它会被传递给生成引擎的各个组件
# Users create one SamplingParams object, which gets passed to various engine components
# ============================================================================
class SamplingParams:
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.sglang.io/backend/sampling_params.html
    for the documentation.
    """

    def __init__(
        self,
        max_new_tokens: int = 128,  # 最大生成token数 (maximum tokens to generate) - 控制输出长度的硬上限
        stop: Optional[Union[str, List[str]]] = None,  # 停止字符串 (stop strings) - 遇到这些文本立即停止生成
        stop_token_ids: Optional[List[int]] = None,  # 停止token ID列表 (stop token IDs) - 遇到这些token立即停止
        stop_regex: Optional[Union[str, List[str]]] = None,  # 停止正则表达式 (stop regex patterns) - 匹配模式时停止
        temperature: float = 1.0,  # 温度参数 (temperature) - 【关键】控制随机性: 0=确定性, >1=更随机/创造性
        top_p: float = 1.0,  # 核采样阈值 (nucleus sampling threshold) - 累积概率达到top_p时截断，保留最可能的词
        top_k: int = -1,  # Top-K采样 (top-k sampling) - 只从概率最高的k个词中采样，-1表示考虑全部词表
        min_p: float = 0.0,  # 最小概率阈值 (minimum probability threshold) - 过滤掉概率低于此值的token
        frequency_penalty: float = 0.0,  # 频率惩罚 (frequency penalty) - 根据token出现次数降低其概率，减少重复
        presence_penalty: float = 0.0,  # 存在惩罚 (presence penalty) - 只要token出现过就惩罚，鼓励话题多样性
        repetition_penalty: float = 1.0,  # 重复惩罚 (repetition penalty) - >1时惩罚已出现的token，防止循环重复
        min_new_tokens: int = 0,  # 最小生成token数 (minimum tokens to generate) - 强制生成至少这么多token
        n: int = 1,  # 并行生成数量 (number of parallel generations) - 一次请求生成n个不同的输出
        json_schema: Optional[str] = None,  # JSON schema约束 (JSON schema constraint) - 强制输出符合JSON格式
        regex: Optional[str] = None,  # 正则表达式约束 (regex constraint) - 强制输出匹配正则模式
        ebnf: Optional[str] = None,  # EBNF语法约束 (EBNF grammar constraint) - 强制输出符合EBNF语法
        structural_tag: Optional[str] = None,  # 结构化标签 (structural tag) - 内部使用的结构化标记
        ignore_eos: bool = False,  # 忽略结束符 (ignore end-of-sequence) - True时即使遇到EOS也继续生成
        skip_special_tokens: bool = True,  # 跳过特殊token (skip special tokens) - 解码时是否跳过<pad>等特殊token
        spaces_between_special_tokens: bool = True,  # 特殊token间加空格 (add spaces between special tokens)
        no_stop_trim: bool = False,  # 不修剪停止符 (don't trim stop strings) - False时会从输出中移除stop string
        custom_params: Optional[Dict[str, Any]] = None,  # 自定义参数字典 (custom parameters dict) - 扩展用途
        stream_interval: Optional[int] = None,  # 流式输出间隔 (streaming interval) - 每隔多少token返回一次中间结果
        logit_bias: Optional[Dict[str, float]] = None,  # Logit偏置 (logit bias) - 手动调整特定token的生成概率
        sampling_seed: Optional[int] = None,  # 采样随机种子 (sampling seed) - 设置后可复现相同的随机输出
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.stop_strs = stop
        if stop_token_ids:
            self.stop_token_ids = set(stop_token_ids)
        else:
            self.stop_token_ids = None
        self.stop_regex_strs = stop_regex
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.min_new_tokens = min_new_tokens
        self.regex = regex
        self.n = n
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.structural_tag = structural_tag
        self.ignore_eos = ignore_eos
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.no_stop_trim = no_stop_trim
        self.custom_params = custom_params
        self.stream_interval = stream_interval
        self.logit_bias = logit_bias
        self.sampling_seed = sampling_seed

        # Process some special cases
        if 0 <= self.temperature < _SAMPLING_EPS:
            # top_k = 1 means greedy sampling
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = TOP_K_ALL  # whole vocabulary

    def verify(self, vocab_size):
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < 1 or self.top_k == -1:
            raise ValueError(
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in [0, 2], got "
                f"{self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in [0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in [0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        if self.logit_bias is not None:
            for token_id in self.logit_bias:
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                        f"{token_id}."
                    )

        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

    def normalize(self, tokenizer):
        # Process stop strings
        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            self.stop_str_max_len = stop_str_max_len

        # Process stop regex strings
        if self.stop_regex_strs is None:
            self.stop_regex_strs = []
            self.stop_regex_max_len = 0
        else:
            if isinstance(self.stop_regex_strs, str):
                self.stop_regex_strs = [self.stop_regex_strs]

            stop_regex_max_len = 0
            for stop_regex in self.stop_regex_strs:
                stop_regex_max_len = max(
                    stop_regex_max_len, get_max_seq_length(stop_regex)
                )

            self.stop_regex_max_len = stop_regex_max_len


# This function gets a strict upperbound on the maximum number of tokens that would need
# to be buffered to match the input regex string
# NOTE: in the worst case, one character that needs to be buffered corresponds to one
# token
def get_max_seq_length(regex_str: str):
    return _max_length_from_subpattern(sre_parse.parse(regex_str))


MAX_LEN = 2**30


def _max_length_from_subpattern(subpattern: sre_parse.SubPattern):
    total = 0
    for token, value in subpattern:
        if token in {
            sre_parse.LITERAL,  # `value` is any one character
            sre_parse.IN,  # Any character within `value`
            sre_parse.ANY,  # "."
        }:
            total += 1
        elif token == sre_parse.SUBPATTERN:
            # EG: (a\d+) ->
            # [(SUBPATTERN,
            #   (1, 0, 0, [(LITERAL, 97),
            #              (MAX_REPEAT, (1, MAXREPEAT, [(IN, [(CATEGORY, CATEGORY_DIGIT)])]))]))]
            _, _, _, inner_subpattern = value
            total += _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.BRANCH:
            _, branches = value
            total += max(_max_length_from_subpattern(branch) for branch in branches)
        elif token in {sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT}:
            _, max_num_repeat, inner_subpattern = value
            if max_num_repeat == sre_parse.MAXREPEAT:
                total += MAX_LEN
            else:
                total += max_num_repeat * _max_length_from_subpattern(inner_subpattern)
        elif token == sre_parse.AT:
            # These are zero-width assertions like ^, $, and \b that don't add to the max
            # length
            total += 0
        else:
            logger.warning(f"Got unhandled regex token: {token}")

            total += MAX_LEN

    return total
