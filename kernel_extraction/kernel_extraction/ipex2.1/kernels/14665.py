

# Original file: ./tacotron2__25_inference_65.5/tacotron2__25_inference_65.5_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/53/c533mfn3qeayocy7p76uonp4gdi436x43pusk4gq3cap6cveaues.py
# Source Nodes: [add, add_1, l_____stack0_____self___attention_layer_memory_layer_linear_layer, tanh], Original ATen: [aten.add, aten.tanh, aten.view]
# add => add_2
# add_1 => add_3
# l_____stack0_____self___attention_layer_memory_layer_linear_layer => view_1
# tanh => tanh_2
triton_poi_fused_add_tanh_view_7 = async_compile.triton('triton_poi_fused_add_tanh_view_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_tanh_view_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_tanh_view_7(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1376256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 128
    x3 = (xindex // 21504)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x1 + (128*x3)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_out_ptr1 + (x0), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 + tmp0
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
    tl.store(in_out_ptr1 + (x0), tmp5, None)
''')
