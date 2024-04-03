

# Original file: ./hf_Reformer__25_inference_65.5/hf_Reformer__25_inference_65.5_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/as/castwotd4ns7hzjsgjqdtzwc4axnj6b5ulwwgbyk2qdqfa4tpvmg.py
# Source Nodes: [exp_1, logsumexp, logsumexp_1, ne, sub_1, where], Original ATen: [aten.exp, aten.logsumexp, aten.ne, aten.sub, aten.where]
# exp_1 => exp_3
# logsumexp => full_default_1
# logsumexp_1 => abs_2, add_12, amax_1, convert_element_type_22, convert_element_type_23, eq_1, exp_2, log_1, sub_5, sum_2, where_2
# ne => ne
# sub_1 => sub_6
# where => where_1
triton_red_fused_exp_logsumexp_ne_sub_where_28 = async_compile.triton('triton_red_fused_exp_logsumexp_ne_sub_where_28', '''
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
    meta={'signature': {0: '*i64', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_exp_logsumexp_ne_sub_where_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_exp_logsumexp_ne_sub_where_28(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x1 = (xindex // 64)
    _tmp12 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr1 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.full([1, 1], 4096, tl.int64)
        tmp2 = tmp0 % tmp1
        tmp3 = tmp2 + tmp1
        tmp4 = tl.where(((tmp2 != 0) & ((tmp2 < 0) != (tmp1 < 0))), tmp3, tmp2)
        tmp6 = tmp4 != tmp5
        tmp8 = -100000.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = triton_helpers.maximum(_tmp12, tmp11)
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = triton_helpers.max2(_tmp12, 1)[:, None]
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp18 = tl.load(in_ptr1 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tl.full([1, 1], 4096, tl.int64)
        tmp15 = tmp0 % tmp14
        tmp16 = tmp15 + tmp14
        tmp17 = tl.where(((tmp15 != 0) & ((tmp15 < 0) != (tmp14 < 0))), tmp16, tmp15)
        tmp19 = tmp17 != tmp18
        tmp21 = -100000.0
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tl.abs(tmp12)
        tmp25 = float("inf")
        tmp26 = tmp24 == tmp25
        tmp27 = 0.0
        tmp28 = tl.where(tmp26, tmp27, tmp12)
        tmp29 = tmp23 - tmp28
        tmp30 = tl.exp(tmp29)
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask, tmp33, _tmp32)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp38 = tl.load(in_ptr1 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp40 = tl.load(in_ptr2 + (r2 + (128*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp34 = tl.full([1, 1], 4096, tl.int64)
        tmp35 = tmp0 % tmp34
        tmp36 = tmp35 + tmp34
        tmp37 = tl.where(((tmp35 != 0) & ((tmp35 < 0) != (tmp34 < 0))), tmp36, tmp35)
        tmp39 = tmp37 != tmp38
        tmp41 = -100000.0
        tmp42 = tl.where(tmp39, tmp40, tmp41)
        tmp43 = tl.log(tmp32)
        tmp44 = tl.abs(tmp12)
        tmp45 = float("inf")
        tmp46 = tmp44 == tmp45
        tmp47 = 0.0
        tmp48 = tl.where(tmp46, tmp47, tmp12)
        tmp49 = tmp43 + tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp42 - tmp50
        tmp52 = tl.exp(tmp51)
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp52, rmask)
''')
