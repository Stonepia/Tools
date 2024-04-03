

# Original file: ./yolov3___60.0/yolov3___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/aq/caqyrueb3qo4vny2lixf5w3hdai62t5iaj3qstjdgudjwxer3hdv.py
# Source Nodes: [l__mod___module_list_103_activation, l__mod___module_list_104], Original ATen: [aten._to_copy, aten._unsafe_index, aten.leaky_relu]
# l__mod___module_list_103_activation => gt_66, mul_275, where_66
# l__mod___module_list_104 => _unsafe_index_1, convert_element_type_346
triton_poi_fused__to_copy__unsafe_index_leaky_relu_28 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_leaky_relu_28', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_leaky_relu_28', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_leaky_relu_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 8192) % 48
    x1 = (xindex // 128) % 64
    x0 = xindex % 128
    x3 = (xindex // 393216)
    x5 = (xindex // 128)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (x0 + (128*tmp14) + (4096*tmp8) + (98304*x3)), None)
    tmp16 = tmp15 > tmp4
    tmp17 = 0.1
    tmp18 = tmp15 * tmp17
    tmp19 = tl.where(tmp16, tmp15, tmp18)
    tmp20 = tmp19.to(tl.float32)
    tl.store(out_ptr0 + (x0 + (384*x5)), tmp20, None)
''')
