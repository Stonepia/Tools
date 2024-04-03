

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/4m/c4mgmfshxf3v4tzcf7jolssq7yf6df7erhelcqw2rqdndlfcvcio.py
# Source Nodes: [exp_1, logsumexp, logsumexp_1, ne, sub_1, where], Original ATen: [aten.exp, aten.logsumexp, aten.ne, aten.sub, aten.where]
# exp_1 => exp_3
# logsumexp => full_default_1
# logsumexp_1 => abs_2, add_12, amax_1, eq_1, exp_2, log_1, sub_5, sum_2, where_2
# ne => ne
# sub_1 => sub_6
# where => where_1
triton_red_fused_exp_logsumexp_ne_sub_where_26 = async_compile.triton('triton_red_fused_exp_logsumexp_ne_sub_where_26', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_exp_logsumexp_ne_sub_where_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_exp_logsumexp_ne_sub_where_26(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x1 = (xindex // 64)
    _tmp11 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr1 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], 4096, tl.int64)
        tmp2 = tmp0 % tmp1
        tmp3 = tmp2 + tmp1
        tmp4 = tl.where(((tmp2 != 0) & ((tmp2 < 0) != (tmp1 < 0))), tmp3, tmp2)
        tmp6 = tmp4 != tmp5
        tmp8 = -100000.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp17 = tl.load(in_ptr1 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp13 = tl.full([1, 1], 4096, tl.int64)
        tmp14 = tmp0 % tmp13
        tmp15 = tmp14 + tmp13
        tmp16 = tl.where(((tmp14 != 0) & ((tmp14 < 0) != (tmp13 < 0))), tmp15, tmp14)
        tmp18 = tmp16 != tmp17
        tmp20 = -100000.0
        tmp21 = tl.where(tmp18, tmp19, tmp20)
        tmp22 = tl.abs(tmp11)
        tmp23 = float("inf")
        tmp24 = tmp22 == tmp23
        tmp25 = 0.0
        tmp26 = tl.where(tmp24, tmp25, tmp11)
        tmp27 = tmp21 - tmp26
        tmp28 = tl.exp(tmp27)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp36 = tl.load(in_ptr1 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp38 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp32 = tl.full([1, 1], 4096, tl.int64)
        tmp33 = tmp0 % tmp32
        tmp34 = tmp33 + tmp32
        tmp35 = tl.where(((tmp33 != 0) & ((tmp33 < 0) != (tmp32 < 0))), tmp34, tmp33)
        tmp37 = tmp35 != tmp36
        tmp39 = -100000.0
        tmp40 = tl.where(tmp37, tmp38, tmp39)
        tmp41 = tl.log(tmp30)
        tmp42 = tl.abs(tmp11)
        tmp43 = float("inf")
        tmp44 = tmp42 == tmp43
        tmp45 = 0.0
        tmp46 = tl.where(tmp44, tmp45, tmp11)
        tmp47 = tmp41 + tmp46
        tmp48 = tmp40 - tmp47
        tmp49 = tl.exp(tmp48)
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp49, rmask)
''')
