

# Original file: ./OPTForCausalLM___60.0/OPTForCausalLM___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/s3/cs33mjzk67ta4meedjxqklg3n66befx32fu6vjo3kv2jc3whsdv5.py
# Source Nodes: [bool_1, masked_fill, masked_fill_1, sub, to_2], Original ATen: [aten._to_copy, aten.masked_fill, aten.rsub]
# bool_1 => convert_element_type_1
# masked_fill => full_default_2, where_1
# masked_fill_1 => full_default_3, where_2
# sub => sub
# to_2 => convert_element_type
triton_poi_fused__to_copy_masked_fill_rsub_0 = async_compile.triton('triton_poi_fused__to_copy_masked_fill_rsub_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_masked_fill_rsub_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_masked_fill_rsub_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2048
    x2 = (xindex // 4194304)
    x1 = (xindex // 2048) % 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*x2)), None, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp1 - tmp0
    tmp3 = (tmp2 != 0)
    tmp4 = -3.4028234663852886e+38
    tmp5 = tl.where(tmp3, tmp4, tmp2)
    tmp6 = (tmp5 != 0)
    tmp7 = x0
    tmp8 = 1 + x1
    tmp9 = tmp7 < tmp8
    tmp10 = 0.0
    tmp11 = tl.where(tmp9, tmp10, tmp4)
    tmp12 = tl.where(tmp6, tmp4, tmp11)
    tl.store(out_ptr0 + (x3), tmp12, None)
''')
