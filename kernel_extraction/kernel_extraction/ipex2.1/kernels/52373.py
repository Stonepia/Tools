

# Original file: ./phlippe_densenet___60.0/phlippe_densenet___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/tj/ctj6j2uxj7xi2jmt2h3ucktleaeu3ee5wekgobhphniqfcde4kcq.py
# Source Nodes: [l__self___output_net_0, l__self___output_net_1, l__self___output_net_2], Original ATen: [aten._adaptive_avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu]
# l__self___output_net_0 => add_103, convert_element_type_209, mul_154, mul_155, sub_51
# l__self___output_net_1 => relu_51
# l__self___output_net_2 => _adaptive_avg_pool2d
triton_poi_fused__adaptive_avg_pool2d__native_batch_norm_legit_no_training_relu_53 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d__native_batch_norm_legit_no_training_relu_53', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d__native_batch_norm_legit_no_training_relu_53', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d__native_batch_norm_legit_no_training_relu_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 184
    x1 = (xindex // 184)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2944*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (184 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (368 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (552 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (736 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (920 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (1104 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (1288 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (1472 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (1656 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (1840 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp21 = tl.load(in_ptr0 + (2024 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp23 = tl.load(in_ptr0 + (2208 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp25 = tl.load(in_ptr0 + (2392 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp27 = tl.load(in_ptr0 + (2576 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp29 = tl.load(in_ptr0 + (2760 + x0 + (2944*x1)), xmask).to(tl.float32)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = 0.0625
    tmp32 = tmp30 * tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''')
