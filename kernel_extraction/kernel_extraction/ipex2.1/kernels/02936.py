

# Original file: ./cm3leon_generate__23_inference_63.3/cm3leon_generate__23_inference_63.3_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/2v/c2vwx5cjqglrhakmhxgwogfirnouyxjsjaekw7ppw543ph47zsoh.py
# Source Nodes: [softmax_1, type_as_3], Original ATen: [aten._softmax, aten._to_copy]
# softmax_1 => div_1, exp_1, sum_2
# type_as_3 => convert_element_type_37
triton_poi_fused__softmax__to_copy_11 = async_compile.triton('triton_poi_fused__softmax__to_copy_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax__to_copy_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__softmax__to_copy_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 2)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr0 + (2*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + (2*x1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.exp(tmp0)
    tmp3 = tl.exp(tmp2)
    tmp5 = tl.exp(tmp4)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp1 / tmp6
    tmp8 = tmp7.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''')
