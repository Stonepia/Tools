

# Original file: ./LayoutLMForSequenceClassification__0_backward_135.1/LayoutLMForSequenceClassification__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/bb/cbbr4ylhew7gvvg6nqpirf7usz3dw2b67c22kpghecv6uejt47f6.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]

triton_red_fused__to_copy_native_layer_norm_backward_select_backward_slice_backward_12 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_backward_select_backward_slice_backward_12', '''
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
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_backward_select_backward_slice_backward_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_backward_select_backward_slice_backward_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex % 512
        r2 = (rindex // 512)
        tmp3 = tl.load(in_ptr0 + (x0 + (768*r2)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp0 = r1
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 0.0
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')