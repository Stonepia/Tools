

# Original file: ./ElectraForCausalLM__0_backward_135.1/ElectraForCausalLM__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/jn/cjnlqqd4z7qcglsoucj7n4vou6uxajm5ghdtg4tl45mlnotu4v2h.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_red_fused__to_copy_native_dropout_backward_native_layer_norm_backward_31 = async_compile.triton('triton_red_fused__to_copy_native_dropout_backward_native_layer_norm_backward_31', '''
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
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_dropout_backward_native_layer_norm_backward_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_native_dropout_backward_native_layer_norm_backward_31(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first')
        tmp7 = tl.load(in_ptr2 + (x0 + (128*r2) + (16384*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 1.1111111111111112
        tmp5 = tmp3 * tmp4
        tmp6 = tmp1 * tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, None)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp13, None)
''')