

# Original file: ./AlbertForQuestionAnswering__0_backward_207.1/AlbertForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/oy/coyt3gyp3dcatzw6gm4x576azd5lqyefhxjdky6uh2fkf3idbtek.py
# Source Nodes: [], Original ATen: [aten.add, aten.native_layer_norm_backward]

triton_red_fused_add_native_layer_norm_backward_13 = async_compile.triton('triton_red_fused_add_native_layer_norm_backward_13', '''
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
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_backward_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_backward_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp3 = tl.load(in_ptr2 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp5 = tl.load(in_ptr3 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp7 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tmp11
        tl.store(out_ptr0 + (r1 + (4096*x0)), tmp8, None)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(out_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last')
        tmp13 = tl.load(in_ptr5 + (r1 + (4096*x0)), None, eviction_policy='evict_last')
        tmp14 = tmp12 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tmp17
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(out_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp23 = tl.load(in_ptr5 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp20 = 4096.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21 - tmp10
        tmp24 = tmp23 * tmp16
        tmp25 = tmp22 - tmp24
        tmp26 = tmp18 * tmp25
        tl.store(out_ptr3 + (r1 + (4096*x0)), tmp26, None)
''')
