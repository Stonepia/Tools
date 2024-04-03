

# Original file: ./BartForConditionalGeneration__35_forward_110.7/BartForConditionalGeneration__35_forward_110.7.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/nh/cnhvdxoj6abcfsoa6wuaa7qjafwimbum2aolisv3zn7cpo2erwsz.py
# Source Nodes: [any_1, isinf, l__self___final_layer_norm], Original ATen: [aten.any, aten.isinf, aten.native_layer_norm]
# any_1 => any_1
# isinf => isinf
# l__self___final_layer_norm => add_6, convert_element_type_7, mul_10, mul_11, sub_2
triton_red_fused_any_isinf_native_layer_norm_7 = async_compile.triton('triton_red_fused_any_isinf_native_layer_norm_7', '''
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_any_isinf_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_any_isinf_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        r2 = (rindex // 1024)
        r1 = rindex % 1024
        tmp0 = tl.load(in_ptr0 + (r3 + (8192*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r2 + (8*x0)), xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r2 + (8*x0)), xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp9 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 - tmp2
        tmp5 = tmp3 * tmp4
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 * tmp7
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp8 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = libdevice.isinf(tmp12).to(tl.int1)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 | tmp14
        _tmp15 = tl.where(xmask, tmp16, _tmp15)
        tl.store(out_ptr0 + (r3 + (8192*x0)), tmp12, xmask)
    tmp15 = triton_helpers.any(_tmp15.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr1 + (x0), tmp15, xmask)
''')
