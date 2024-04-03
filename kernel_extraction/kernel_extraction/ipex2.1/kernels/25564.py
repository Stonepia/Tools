

# Original file: ./sam___60.0/sam___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/kd/ckddqtaxlef4vjyyh4qq2volujpr2puogtvuw72btao6nd4zybme.py
# Source Nodes: [softmax_7], Original ATen: [aten._softmax]
# softmax_7 => amax_7, convert_element_type_136, convert_element_type_137, div_8, exp_7, sub_39, sum_8
triton_red_fused__softmax_28 = async_compile.triton('triton_red_fused__softmax_28', '''
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
    size_hints=[65536, 4096],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_28(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 4096
    x1 = (xindex // 4096)
    _tmp7 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x3)), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + ((64*(x0 % 64)) + (4096*x1) + (65536*(x0 // 64)) + (r2 // 64)), None, eviction_policy='evict_last').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + ((64*(x0 // 64)) + (4096*x1) + (65536*(x0 % 64)) + (r2 % 64)), None, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = triton_helpers.maximum(_tmp7, tmp6)
        _tmp7 = tmp8
    tmp7 = triton_helpers.max2(_tmp7, 1)[:, None]
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (r2 + (4096*x3)), None, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr1 + ((64*(x0 % 64)) + (4096*x1) + (65536*(x0 // 64)) + (r2 // 64)), None, eviction_policy='evict_last').to(tl.float32)
        tmp12 = tl.load(in_ptr2 + ((64*(x0 // 64)) + (4096*x1) + (65536*(x0 % 64)) + (r2 % 64)), None, eviction_policy='evict_last').to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp7
        tmp16 = tl.exp(tmp15)
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tmp19
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp20 = tl.load(in_ptr0 + (r2 + (4096*x3)), None, eviction_policy='evict_first').to(tl.float32)
        tmp21 = tl.load(in_ptr1 + ((64*(x0 % 64)) + (4096*x1) + (65536*(x0 // 64)) + (r2 // 64)), None, eviction_policy='evict_first').to(tl.float32)
        tmp23 = tl.load(in_ptr2 + ((64*(x0 // 64)) + (4096*x1) + (65536*(x0 % 64)) + (r2 % 64)), None, eviction_policy='evict_first').to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp25 - tmp7
        tmp27 = tl.exp(tmp26)
        tmp28 = tmp27 / tmp18
        tmp29 = tmp28.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (4096*x3)), tmp29, None)
''')
