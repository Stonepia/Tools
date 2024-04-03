

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/io/ciobnz5lmxcd255ckksng6wam6uegsn3dleog45v6yoanfn5man6.py
# Source Nodes: [avg_pool2d_3, leaky_relu_7], Original ATen: [aten.avg_pool2d, aten.leaky_relu]
# avg_pool2d_3 => avg_pool2d_3
# leaky_relu_7 => convert_element_type_31, convert_element_type_32, gt_7, mul_7, where_7
triton_poi_fused_avg_pool2d_leaky_relu_9 = async_compile.triton('triton_poi_fused_avg_pool2d_leaky_relu_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_leaky_relu_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_leaky_relu_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 743424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 22
    x2 = (xindex // 5632)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (22528*x2)), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (22528*x2)), None).to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (11264 + x0 + (512*x1) + (22528*x2)), None).to(tl.float32)
    tmp22 = tl.load(in_ptr0 + (11520 + x0 + (512*x1) + (22528*x2)), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp4 = 0.1
    tmp5 = tmp1 * tmp4
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 > tmp2
    tmp11 = tmp9 * tmp4
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 + tmp7
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 > tmp2
    tmp18 = tmp16 * tmp4
    tmp19 = tl.where(tmp17, tmp16, tmp18)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 + tmp14
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 > tmp2
    tmp25 = tmp23 * tmp4
    tmp26 = tl.where(tmp24, tmp23, tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27 + tmp21
    tmp29 = 0.25
    tmp30 = tmp28 * tmp29
    tl.store(out_ptr0 + (x3), tmp30, None)
''')
