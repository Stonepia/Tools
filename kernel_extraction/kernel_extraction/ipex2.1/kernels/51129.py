

# Original file: ./hf_Longformer__22_inference_62.2/hf_Longformer__22_inference_62.2_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/7r/c7rhdmwrkwe7d2zojvrqjhzn7y757gyytianz4boohm3bxnlnjb7.py
# Source Nodes: [bool_1, full_like, where], Original ATen: [aten._to_copy, aten.full_like, aten.where]
# bool_1 => convert_element_type_9
# full_like => full_default_2
# where => where_1
triton_poi_fused__to_copy_full_like_where_7 = async_compile.triton('triton_poi_fused__to_copy_full_like_where_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_full_like_where_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_full_like_where_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 789504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 257
    x1 = (xindex // 257) % 256
    x3 = (xindex // 257)
    x2 = (xindex // 65792)
    x4 = xindex
    tmp9 = tl.load(in_ptr0 + (x0 + (513*x3)), xmask).to(tl.float32)
    tmp26 = tl.load(in_ptr2 + (x0 + (513*x1) + (2101248*x2)), xmask).to(tl.float32)
    tmp0 = (-255) + x0 + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 <= tmp1
    tmp3 = 1.0
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = (tmp5 != 0)
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp7 == tmp7
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp1 >= tmp10
    tmp12 = x0
    tmp13 = tl.full([1], 256, tl.int64)
    tmp14 = tmp12 < tmp13
    tmp15 = tmp14 & tmp11
    tmp16 = (((-131584) + x0 + (513*x1) + (3939840*x2)) // 512) % 513
    tmp17 = tl.full([1], 512, tl.int64)
    tmp18 = tmp16 < tmp17
    tmp19 = tmp18 & tmp15
    tmp20 = tl.load(in_ptr1 + ((512*((((-131584) + x0 + (513*x1) + (3939840*x2)) // 512) % 513)) + (262144*((((-131584) + x0 + (513*x1) + (3939840*x2)) // 262656) % 180)) + ((x0 + (513*x1)) % 512)), tmp19 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp19, tmp20, 0.0)
    tmp22 = tl.where(tmp15, tmp21, 0.0)
    tmp23 = tl.load(in_ptr2 + (x0 + (513*x1) + (2101248*x2)), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp14, tmp22, tmp23)
    tmp25 = tl.where(tmp11, tmp24, 0.0)
    tmp27 = tl.where(tmp11, tmp25, tmp26)
    tmp28 = tl.where(tmp8, tmp9, tmp27)
    tmp29 = float("-inf")
    tmp30 = tl.where(tmp6, tmp29, tmp28)
    tl.store(out_ptr0 + (x4), tmp30, xmask)
''')
