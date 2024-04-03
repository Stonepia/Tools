

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/sj/csjrwa2lrnxgvax2vjh4ntbwiy4n7gvfncxsjdrmrztniytohaxl.py
# Source Nodes: [einsum_5, einsum_6], Original ATen: [aten._to_copy]
# einsum_5 => convert_element_type_11
# einsum_6 => convert_element_type_12
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x1 = (xindex // 64) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + (64*(x1 % 16))), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0 + (64*(x1 % 16))), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp1 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp4, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
''')
