

# Original file: ./LearningToPaint___60.0/LearningToPaint___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/lk/clk7yvfyd57vbyry7mbyntawsrpvajjbe2hli65zfu3b6of5ftlr.py
# Source Nodes: [avg_pool2d], Original ATen: [aten.avg_pool2d]
# avg_pool2d => avg_pool2d
triton_poi_fused_avg_pool2d_1 = async_compile.triton('triton_poi_fused_avg_pool2d_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (8192*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (512 + x0 + (8192*x1)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (1024 + x0 + (8192*x1)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (1536 + x0 + (8192*x1)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (2048 + x0 + (8192*x1)), None).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (2560 + x0 + (8192*x1)), None).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (3072 + x0 + (8192*x1)), None).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (3584 + x0 + (8192*x1)), None).to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (4096 + x0 + (8192*x1)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (4608 + x0 + (8192*x1)), None).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (5120 + x0 + (8192*x1)), None).to(tl.float32)
    tmp21 = tl.load(in_ptr0 + (5632 + x0 + (8192*x1)), None).to(tl.float32)
    tmp23 = tl.load(in_ptr0 + (6144 + x0 + (8192*x1)), None).to(tl.float32)
    tmp25 = tl.load(in_ptr0 + (6656 + x0 + (8192*x1)), None).to(tl.float32)
    tmp27 = tl.load(in_ptr0 + (7168 + x0 + (8192*x1)), None).to(tl.float32)
    tmp29 = tl.load(in_ptr0 + (7680 + x0 + (8192*x1)), None).to(tl.float32)
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
    tl.store(out_ptr0 + (x2), tmp32, None)
''')
