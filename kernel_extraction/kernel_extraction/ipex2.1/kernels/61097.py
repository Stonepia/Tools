

# Original file: ./timm_efficientnet___60.0/timm_efficientnet___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/iq/ciqvidsp27ysyyxkwyy4m3y7ufetv2wmvwim7r56xyadci7w6ana.py
# Source Nodes: [batch_norm_4, getattr_getattr_l__self___blocks___1_____0___bn2_act, mean_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.silu]
# batch_norm_4 => add_9, mul_18, mul_19, sub_4
# getattr_getattr_l__self___blocks___1_____0___bn2_act => convert_element_type_34, mul_20, sigmoid_5
# mean_1 => mean_1
triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8 = async_compile.triton('triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8', '''
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
    size_hints=[8192, 4096],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_mean_silu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 3136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 96
    x1 = (xindex // 96)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (96*r2) + (301056*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp7 = tmp5 * tmp6
        tmp8 = tmp7 + tmp2
        tmp9 = tl.sigmoid(tmp8)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp16 = 3136.0
    tmp17 = tmp14 / tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(out_ptr1 + (x3), tmp18, None)
''')