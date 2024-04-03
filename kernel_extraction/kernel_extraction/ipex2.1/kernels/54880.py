

# Original file: ./AllenaiLongformerBase__22_forward_137.2/AllenaiLongformerBase__22_forward_137.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/zv/czv3vgj4sr2lpy4wgeslksez33dsfubsxx4uyhjch6aondenr5fm.py
# Source Nodes: [bool_1, full_like, where], Original ATen: [aten._to_copy, aten.full_like, aten.where]
# bool_1 => convert_element_type
# full_like => full_default_2
# where => where_1
triton_poi_fused__to_copy_full_like_where_9 = async_compile.triton('triton_poi_fused__to_copy_full_like_where_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_full_like_where_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_full_like_where_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3158016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex % 65792
    x0 = xindex % 257
    x6 = (xindex // 257)
    x1 = (xindex // 257) % 256
    x4 = (xindex // 65792)
    x2 = (xindex // 65792) % 12
    x3 = (xindex // 789504)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (x5), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x0 + (513*x6)), None).to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x0 + (513*x1) + (525312*x4)), None).to(tl.float32)
    tmp1 = (tmp0 != 0)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp2 == tmp2
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = x0
    tmp9 = tl.full([1], 256, tl.int64)
    tmp10 = tmp8 < tmp9
    tmp11 = tmp10 & tmp7
    tmp12 = (((-131584) + x0 + (513*x1) + (787968*x4)) // 512) % 513
    tmp13 = tl.full([1], 512, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14 & tmp11
    tmp16 = tl.load(in_ptr2 + ((512*((((-131584) + x0 + (513*x1) + (787968*x2) + (9455616*x3)) // 512) % 513)) + (262144*((((-131584) + x0 + (513*x1) + (787968*x2) + (9455616*x3)) // 262656) % 144)) + ((x0 + (513*x1)) % 512)), tmp15, other=0.0).to(tl.float32)
    tmp17 = tl.where(tmp15, tmp16, 0.0)
    tmp18 = tl.where(tmp11, tmp17, 0.0)
    tmp19 = tl.load(in_ptr3 + (x0 + (513*x1) + (525312*x4)), tmp7, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp10, tmp18, tmp19)
    tmp21 = tl.where(tmp7, tmp20, 0.0)
    tmp23 = tl.where(tmp7, tmp21, tmp22)
    tmp24 = tl.where(tmp3, tmp4, tmp23)
    tmp25 = float("-inf")
    tmp26 = tl.where(tmp1, tmp25, tmp24)
    tl.store(out_ptr0 + (x7), tmp26, None)
''')
