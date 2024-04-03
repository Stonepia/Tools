

# Original file: ./demucs__21_inference_61.1/demucs__21_inference_61.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/jn/cjnu4sj3loflecsjicxirbbft63wtknzzwa23ir6jqgwnto4mfkv.py
# Source Nodes: [l__self___encoder_0_0, l__self___encoder_0_1, l__self___encoder_0_2, l__self___encoder_0_3], Original ATen: [aten._to_copy, aten.convolution, aten.glu, aten.relu]
# l__self___encoder_0_0 => convert_element_type, convolution
# l__self___encoder_0_1 => relu
# l__self___encoder_0_2 => convolution_1
# l__self___encoder_0_3 => glu
triton_poi_fused__to_copy_convolution_glu_relu_2 = async_compile.triton('triton_poi_fused__to_copy_convolution_glu_relu_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_convolution_glu_relu_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_convolution_glu_relu_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48996352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6124544)
    x3 = xindex % 6124544
    x1 = (xindex // 95696) % 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (12249088*x2)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (6124544 + x3 + (12249088*x2)), None).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (64 + x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tl.store(out_ptr0 + (x4), tmp7, None)
''')
