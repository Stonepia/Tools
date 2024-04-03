

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/gl/cglolgjvyxdzteelywksk7vwz2tu3fzf5wpzsj65aoi64hemomva.py
# Source Nodes: [einsum_4], Original ATen: [aten._to_copy]
# einsum_4 => convert_element_type_9
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (x0), None)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.1111111111111112
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp10, None)
''')
