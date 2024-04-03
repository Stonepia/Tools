

# Original file: ./OPTForCausalLM__58_backward_175.15/OPTForCausalLM__58_backward_175.15_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/rj/crj5xymo2el26um3cyqsencx4t7km32eiugzco4rh34iyqhmlj7g.py
# Source Nodes: [], Original ATen: [aten.add, aten.slice_backward]

triton_poi_fused_add_slice_backward_3 = async_compile.triton('triton_poi_fused_add_slice_backward_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp16', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_slice_backward_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_slice_backward_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205914112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 50272) % 2048
    x2 = (xindex // 102957056)
    x4 = xindex % 102957056
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp8 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp1 = x1
    tmp2 = tl.full([1], 2047, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.load(in_ptr1 + (x4 + (102906784*x2)), tmp3, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x1 + (2047*x2)), tmp3, eviction_policy='evict_last')
    tmp10 = tmp7 / tmp9
    tmp11 = 0.0
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tmp13 = tmp4 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.load(in_ptr5 + (x4 + (102906784*x2)), tmp3, other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.exp(tmp16)
    tmp18 = tl.load(in_ptr6 + (x1 + (2047*x2)), tmp3, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp14 - tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.where(tmp3, tmp21, 0.0)
    tmp23 = tl.where(tmp3, tmp22, tmp11)
    tmp24 = tmp0 + tmp23
    tl.store(out_ptr0 + (x3), tmp24, None)
''')
