use js_sys::{Array, Function, Object, Promise, Reflect, Uint16Array, Uint32Array, Uint8Array};
use std::cell::RefCell;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

// WebGPU constants
const BUF_MAP_READ: u32 = 1;
const BUF_COPY_SRC: u32 = 4;
const BUF_COPY_DST: u32 = 8;
const BUF_STORAGE: u32 = 128;
const SHADER_COMPUTE: u32 = 4;

// Triangle blocking constants
const B: u32 = 256;
const S: u32 = 128;

// ── JS interop helpers ──────────────────────────────────────────────────────

fn js_get(obj: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    Reflect::get(obj, &key.into())
}

fn js_obj(entries: &[(&str, JsValue)]) -> Object {
    let obj = Object::new();
    for (k, v) in entries {
        Reflect::set(&obj, &(*k).into(), v).unwrap();
    }
    obj
}

fn js_call0(obj: &JsValue, method: &str) -> Result<JsValue, JsValue> {
    js_get(obj, method)?.dyn_into::<Function>()?.call0(obj)
}

fn js_call1(obj: &JsValue, method: &str, a: &JsValue) -> Result<JsValue, JsValue> {
    js_get(obj, method)?.dyn_into::<Function>()?.call1(obj, a)
}

fn js_call2(obj: &JsValue, method: &str, a: &JsValue, b: &JsValue) -> Result<JsValue, JsValue> {
    js_get(obj, method)?
        .dyn_into::<Function>()?
        .call2(obj, a, b)
}

async fn js_await(val: JsValue) -> Result<JsValue, JsValue> {
    JsFuture::from(val.dyn_into::<Promise>()?).await
}

fn create_buffer(device: &JsValue, size: u32, usage: u32) -> Result<JsValue, JsValue> {
    js_call1(
        device,
        "createBuffer",
        &js_obj(&[("size", (size as f64).into()), ("usage", usage.into())]),
    )
}

fn write_buffer_u8(queue: &JsValue, buffer: &JsValue, data: &[u8]) -> Result<(), JsValue> {
    let js_data: JsValue = Uint8Array::from(data).into();
    Reflect::apply(
        &js_get(queue, "writeBuffer")?.dyn_into::<Function>()?,
        queue,
        &Array::of3(buffer, &0.into(), &js_data),
    )?;
    Ok(())
}

fn write_buffer_u32(queue: &JsValue, buffer: &JsValue, data: &[u32]) -> Result<(), JsValue> {
    let js_data: JsValue = Uint32Array::from(data).into();
    Reflect::apply(
        &js_get(queue, "writeBuffer")?.dyn_into::<Function>()?,
        queue,
        &Array::of3(buffer, &0.into(), &js_data),
    )?;
    Ok(())
}

fn create_shader(device: &JsValue, code: &str) -> Result<JsValue, JsValue> {
    js_call1(
        device,
        "createShaderModule",
        &js_obj(&[("code", code.into())]),
    )
}

fn create_pipeline(
    device: &JsValue,
    layout: &JsValue,
    module: &JsValue,
    entry: &str,
) -> Result<JsValue, JsValue> {
    let compute = js_obj(&[("module", module.clone()), ("entryPoint", entry.into())]);
    let desc = js_obj(&[("layout", layout.clone()), ("compute", compute.into())]);
    js_call1(device, "createComputePipeline", &desc)
}

fn create_storage_bgl(
    device: &JsValue,
    count: u32,
    read_only_mask: u32,
) -> Result<JsValue, JsValue> {
    let entries = Array::new();
    for i in 0..count {
        let is_ro = (read_only_mask >> i) & 1 == 1;
        let buf_desc = js_obj(&[(
            "type",
            if is_ro {
                "read-only-storage"
            } else {
                "storage"
            }
            .into(),
        )]);
        entries.push(&js_obj(&[
            ("binding", i.into()),
            ("visibility", SHADER_COMPUTE.into()),
            ("buffer", buf_desc.into()),
        ]));
    }
    js_call1(
        device,
        "createBindGroupLayout",
        &js_obj(&[("entries", entries.into())]),
    )
}

fn create_bind_group(
    device: &JsValue,
    layout: &JsValue,
    buffers: &[&JsValue],
) -> Result<JsValue, JsValue> {
    let entries = Array::new();
    for (i, buf) in buffers.iter().enumerate() {
        let resource = js_obj(&[("buffer", (*buf).clone())]);
        entries.push(&js_obj(&[
            ("binding", (i as u32).into()),
            ("resource", resource.into()),
        ]));
    }
    js_call1(
        device,
        "createBindGroup",
        &js_obj(&[("layout", layout.clone()), ("entries", entries.into())]),
    )
}

fn dispatch(
    encoder: &JsValue,
    pipeline: &JsValue,
    bind_group: &JsValue,
    workgroups: u32,
) -> Result<(), JsValue> {
    let pass = js_call1(encoder, "beginComputePass", &Object::new())?;
    js_call1(&pass, "setPipeline", pipeline)?;
    js_call2(&pass, "setBindGroup", &0.into(), bind_group)?;
    js_call1(&pass, "dispatchWorkgroups", &workgroups.into())?;
    js_call0(&pass, "end")?;
    Ok(())
}

// ── WGSL shaders ────────────────────────────────────────────────────────────

fn lab_conversion_shader(n: u32) -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> rgba: array<u32>;
@group(0) @binding(1) var<storage, read_write> lab_out: array<vec4f>;

fn srgb_to_linear(c: f32) -> f32 {{
  if (c <= 0.04045) {{ return c / 12.92; }}
  return pow((c + 0.055) / 1.055, 2.4);
}}

fn cbrt_signed(x: f32) -> f32 {{
  return sign(x) * pow(abs(x), 1.0 / 3.0);
}}

@compute @workgroup_size(256)
fn convert(@builtin(global_invocation_id) gid: vec3u) {{
  let idx = gid.x;
  if (idx >= {n}u) {{ return; }}
  let p = rgba[idx];
  let lr = srgb_to_linear(f32(p & 0xFFu) / 255.0);
  let lg = srgb_to_linear(f32((p >> 8u) & 0xFFu) / 255.0);
  let lb = srgb_to_linear(f32((p >> 16u) & 0xFFu) / 255.0);

  // Linear Display P3 → LMS (precomputed M_xyz_to_lms · M_p3_to_xyz).
  let lms_l = 0.4813371*lr + 0.4620734*lg + 0.0565038*lb;
  let lms_m = 0.2288498*lr + 0.6532486*lg + 0.1179665*lb;
  let lms_s = 0.0839833*lr + 0.2242765*lg + 0.6922382*lb;

  let l_ = cbrt_signed(lms_l);
  let m_ = cbrt_signed(lms_m);
  let s_ = cbrt_signed(lms_s);

  // OKLab, scaled to (L*100, a*400, b*400) so distances line up with
  // the historical CIELAB-shaped energy magnitudes.
  let big_l = (0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_) * 100.0;
  let aa    = (1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_) * 400.0;
  let bb    = (0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_) * 400.0;
  lab_out[idx] = vec4f(big_l, aa, bb, 0.0);
}}
"#
    )
}

fn seam_carving_shader(mw: u32, mh: u32) -> String {
    format!(
        r#"
const MW: u32 = {mw}u;
const MH: u32 = {mh}u;
const BB: u32 = {B}u;
const SS: u32 = {S}u;

@group(0) @binding(0) var<storage, read_write> lab: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dp: array<f32>;
@group(0) @binding(2) var<storage, read_write> dirs: array<u32>;
@group(0) @binding(3) var<storage, read_write> seam_buf: array<u32>;
@group(0) @binding(4) var<storage, read_write> col_map: array<u32>;
@group(0) @binding(5) var<storage, read_write> order_map: array<u32>;
@group(0) @binding(6) var<storage, read_write> params: array<u32>;

fn ld(a: vec4f, b: vec4f) -> f32 {{ let d = a.xyz - b.xyz; return sqrt(dot(d, d)); }}

fn fwd(i: u32, j: u32, w: u32) -> vec3f {{
  var gl = lab[i*MW+j]; if (j > 0u) {{ gl = lab[i*MW+j-1u]; }}
  var gr = lab[i*MW+j]; if (j+1u < w) {{ gr = lab[i*MW+j+1u]; }}
  let ga = lab[(i-1u)*MW+j];
  let cU = ld(gr, gl);
  return vec3f(cU + ld(ga, gl), cU, cU + ld(ga, gr));
}}

fn dp_step(i: u32, j: u32, w: u32) {{
  let e = fwd(i, j, w);
  var best = dp[(i-1u)*MW+j] + e.y; var dir = 0u;
  if (j > 0u) {{ let v = dp[(i-1u)*MW+j-1u]+e.x; if (v < best) {{ best=v; dir=1u; }} }}
  if (j+1u < w) {{ let v = dp[(i-1u)*MW+j+1u]+e.z; if (v < best) {{ best=v; dir=2u; }} }}
  dp[i*MW+j] = best; dirs[i*MW+j] = dir;
}}

@compute @workgroup_size({B})
fn dp_down(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {{
  let tri=wg.x; let strip=params[1]; let jl=lid.x; let w=params[0];
  let col0=tri*BB; let row0=strip*SS; let ntri=(w+BB-1u)/BB;
  for (var k=0u; k<SS; k++) {{
    let i=row0+k;
    var lft=k; if (tri==0u) {{ lft=0u; }}
    var rgt=BB-k; if (tri>=ntri-1u) {{ rgt=BB; }}
    let j=col0+jl;
    if ((i<MH)&&(jl>=lft)&&(jl<rgt)&&(j<w)) {{
      if (i==0u) {{
        var gl=lab[j]; if (j>0u) {{ gl=lab[j-1u]; }}
        var gr=lab[j]; if (j+1u<w) {{ gr=lab[j+1u]; }}
        dp[j]=ld(gr,gl); dirs[j]=0u;
      }} else {{ dp_step(i,j,w); }}
    }}
    storageBarrier();
  }}
}}

@compute @workgroup_size({B})
fn dp_up(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {{
  let gap=wg.x; let strip=params[1]; let jl=lid.x; let w=params[0];
  let center=(gap+1u)*BB; let row0=strip*SS;
  for (var k=0u; k<SS; k++) {{
    let i=row0+k; var is_active=false; var j=0u;
    if (k>0u && center>=k && jl<2u*k) {{
      j=center-k+jl; is_active=(i<MH)&&(i>0u)&&(j<w);
    }}
    if (is_active) {{ dp_step(i,j,w); }}
    storageBarrier();
  }}
}}

@compute @workgroup_size(1)
fn backtrack() {{
  let w=params[0]; var mv=dp[(MH-1u)*MW]; var mj=0u;
  for (var j=1u; j<w; j++) {{ let v=dp[(MH-1u)*MW+j]; if (v<mv) {{ mv=v; mj=j; }} }}
  seam_buf[MH-1u]=mj; var cj=mj;
  for (var ii=1u; ii<MH; ii++) {{
    let i=MH-1u-ii; let d=dirs[(i+1u)*MW+cj];
    if (d==1u) {{ cj=cj-1u; }} else if (d==2u) {{ cj=cj+1u; }}
    seam_buf[i]=cj;
  }}
}}

@compute @workgroup_size(256)
fn compact(@builtin(global_invocation_id) gid: vec3u) {{
  let row=gid.x; let w=params[0]; if (row>=MH) {{ return; }}
  let sc=seam_buf[row]; let oc=col_map[row*MW+sc];
  order_map[row*MW+oc]=w;
  for (var j=sc; j+1u<w; j++) {{
    lab[row*MW+j]=lab[row*MW+j+1u];
    col_map[row*MW+j]=col_map[row*MW+j+1u];
  }}
}}

@compute @workgroup_size(1) fn dec_width() {{ params[0]=params[0]-1u; }}
@compute @workgroup_size(1) fn reset_strip() {{ params[1]=0u; }}
@compute @workgroup_size(1) fn inc_strip() {{ params[1]=params[1]+1u; }}
"#
    )
}

// ── Main GPU entry point ────────────────────────────────────────────────────

#[wasm_bindgen]
pub async fn precompute_seam_order_gpu(
    image_data: &[u8],
    width: u32,
    height: u32,
    direction: u32,
) -> Result<Vec<u32>, JsValue> {
    let (w, h, data) = if direction == 0 {
        (width, height, image_data.to_vec())
    } else {
        let mut t = vec![0u8; image_data.len()];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let src = (y * width as usize + x) * 4;
                let dst = (x * height as usize + y) * 4;
                t[dst..dst + 4].copy_from_slice(&image_data[src..src + 4]);
            }
        }
        (height, width, t)
    };

    let order = gpu_seam_carve(&data, w, h).await?;

    if direction == 0 {
        Ok(order)
    } else {
        let mut result = vec![0u32; (width * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                result[y * width as usize + x] = order[x * height as usize + y];
            }
        }
        Ok(result)
    }
}

#[wasm_bindgen]
pub fn is_webgpu_available() -> bool {
    let Ok(nav) = js_get(&js_sys::global(), "navigator") else {
        return false;
    };
    let Ok(gpu) = js_get(&nav, "gpu") else {
        return false;
    };
    !gpu.is_undefined() && !gpu.is_null()
}

async fn gpu_seam_carve(image_data: &[u8], w: u32, h: u32) -> Result<Vec<u32>, JsValue> {
    let n = w * h;
    let n_down = w.div_ceil(B);
    let n_up = if n_down > 0 { n_down - 1 } else { 0 };
    let n_strips = h.div_ceil(S);

    // ── Get GPU device ──────────────────────────────────────────────────
    let nav = js_get(&js_sys::global(), "navigator")?;
    let gpu = js_get(&nav, "gpu")?;
    if gpu.is_undefined() || gpu.is_null() {
        return Err("WebGPU not available".into());
    }
    let adapter = js_await(js_call0(&gpu, "requestAdapter")?).await?;
    if adapter.is_null() || adapter.is_undefined() {
        return Err("No GPU adapter".into());
    }
    // Request the adapter's max storage buffer binding size (can exceed u32, keep as f64)
    let max_storage = js_get(&js_get(&adapter, "limits")?, "maxStorageBufferBindingSize")?
        .as_f64()
        .unwrap_or(134217728.0);

    let required_limits = js_obj(&[(
        "maxStorageBufferBindingSize",
        JsValue::from_f64(max_storage),
    )]);
    let device_desc = js_obj(&[("requiredLimits", required_limits.into())]);
    let device = js_await(js_call1(&adapter, "requestDevice", &device_desc)?).await?;
    let queue = js_get(&device, "queue")?;

    // ── Create shaders + pipelines ──────────────────────────────────────
    let lab_mod = create_shader(&device, &lab_conversion_shader(n))?;
    let seam_mod = create_shader(&device, &seam_carving_shader(w, h))?;

    // LAB conversion: binding 0 = read-only, binding 1 = read-write
    let lab_bgl = create_storage_bgl(&device, 2, 0b01)?;
    let lab_pl = {
        let layouts = Array::of1(&lab_bgl);
        js_call1(
            &device,
            "createPipelineLayout",
            &js_obj(&[("bindGroupLayouts", layouts.into())]),
        )?
    };
    let lab_pipe = create_pipeline(&device, &lab_pl, &lab_mod, "convert")?;

    // Seam carving: 7 storage bindings, all read-write
    let seam_bgl = create_storage_bgl(&device, 7, 0)?;
    let seam_pl = {
        let layouts = Array::of1(&seam_bgl);
        js_call1(
            &device,
            "createPipelineLayout",
            &js_obj(&[("bindGroupLayouts", layouts.into())]),
        )?
    };

    let p = |entry: &str| create_pipeline(&device, &seam_pl, &seam_mod, entry);
    let down_p = p("dp_down")?;
    let up_p = p("dp_up")?;
    let bt_p = p("backtrack")?;
    let comp_p = p("compact")?;
    let dec_p = p("dec_width")?;
    let rst_p = p("reset_strip")?;
    let inc_p = p("inc_strip")?;

    // ── Create buffers ──────────────────────────────────────────────────
    let rgba_b = create_buffer(&device, n * 4, BUF_STORAGE | BUF_COPY_DST)?;
    let lab_b = create_buffer(&device, n * 16, BUF_STORAGE)?; // vec4f = 16 bytes
    let dp_b = create_buffer(&device, n * 4, BUF_STORAGE)?;
    let dirs_b = create_buffer(&device, n * 4, BUF_STORAGE)?;
    let seam_b = create_buffer(&device, h * 4, BUF_STORAGE)?;
    let cm_b = create_buffer(&device, n * 4, BUF_STORAGE | BUF_COPY_DST)?;
    let order_b = create_buffer(&device, n * 4, BUF_STORAGE | BUF_COPY_SRC)?;
    let params_b = create_buffer(&device, 8, BUF_STORAGE | BUF_COPY_DST)?;
    let read_b = create_buffer(&device, n * 4, BUF_MAP_READ | BUF_COPY_DST)?;

    // ── Upload data ─────────────────────────────────────────────────────
    write_buffer_u8(&queue, &rgba_b, image_data)?;

    let init_cm: Vec<u32> = (0..h).flat_map(|_| 0..w).collect();
    write_buffer_u32(&queue, &cm_b, &init_cm)?;
    write_buffer_u32(&queue, &params_b, &[w, 0])?;

    // ── LAB conversion ──────────────────────────────────────────────────
    let lab_bg = create_bind_group(&device, &lab_bgl, &[&rgba_b, &lab_b])?;
    let enc = js_call1(&device, "createCommandEncoder", &Object::new())?;
    dispatch(&enc, &lab_pipe, &lab_bg, n.div_ceil(256))?;
    let cmd_buf = js_call0(&enc, "finish")?;
    let cmds = Array::of1(&cmd_buf);
    js_call1(&queue, "submit", &cmds)?;

    // ── Seam carving loop ───────────────────────────────────────────────
    let seam_bg = create_bind_group(
        &device,
        &seam_bgl,
        &[&lab_b, &dp_b, &dirs_b, &seam_b, &cm_b, &order_b, &params_b],
    )?;

    let compact_wg = h.div_ceil(256);
    let batch_size = 50u32;
    let total_seams = w - 1;

    let mut seam0 = 0u32;
    while seam0 < total_seams {
        let batch_end = (seam0 + batch_size).min(total_seams);
        let enc = js_call1(&device, "createCommandEncoder", &Object::new())?;

        for _ in seam0..batch_end {
            // Reset strip counter
            dispatch(&enc, &rst_p, &seam_bg, 1)?;

            // DP fill: strip by strip with triangular blocking
            for _ in 0..n_strips {
                dispatch(&enc, &down_p, &seam_bg, n_down)?;
                if n_up > 0 {
                    dispatch(&enc, &up_p, &seam_bg, n_up)?;
                }
                dispatch(&enc, &inc_p, &seam_bg, 1)?;
            }

            // Backtrack + compact + decrement width
            dispatch(&enc, &bt_p, &seam_bg, 1)?;
            dispatch(&enc, &comp_p, &seam_bg, compact_wg)?;
            dispatch(&enc, &dec_p, &seam_bg, 1)?;
        }

        // Copy result on last batch
        if batch_end >= total_seams {
            let copy_fn: Function = js_get(&enc, "copyBufferToBuffer")?.dyn_into()?;
            let args = Array::of5(&order_b, &0.into(), &read_b, &0.into(), &(n * 4).into());
            Reflect::apply(&copy_fn, &enc, &args)?;
        }

        let cmd_buf = js_call0(&enc, "finish")?;
        js_call1(&queue, "submit", &Array::of1(&cmd_buf))?;

        // Yield periodically to keep event loop responsive
        if batch_end < total_seams && (batch_end % 200 < batch_size) {
            js_await(js_call0(&queue, "onSubmittedWorkDone")?).await?;
        }

        seam0 = batch_end;
    }

    // ── Read back results ───────────────────────────────────────────────
    js_await(js_call0(&queue, "onSubmittedWorkDone")?).await?;

    // buffer.mapAsync(GPUMapMode.READ)
    js_await(js_call1(&read_b, "mapAsync", &1.into())?).await?;

    let mapped = js_call0(&read_b, "getMappedRange")?;
    let raw_order = Uint32Array::new(&mapped);
    let raw: Vec<u32> = raw_order.to_vec();
    js_call0(&read_b, "unmap")?;

    // Convert from GPU convention (order=W → first removed, 0 → surviving)
    // to our convention (order=1 → first removed, W → surviving)
    let mut order = vec![0u32; n as usize];
    for i in 0..n as usize {
        let v = raw[i];
        order[i] = if v == 0 { w } else { w + 1 - v };
    }

    // ── Cleanup ─────────────────────────────────────────────────────────
    for buf in [
        &rgba_b, &lab_b, &dp_b, &dirs_b, &seam_b, &cm_b, &order_b, &params_b, &read_b,
    ] {
        let _ = js_call0(buf, "destroy");
    }

    Ok(order)
}

// ── Persistent WebGPU rendering context ─────────────────────────────────────
//
// Owns the entire live-edit pipeline: source upload → remap compute →
// carve compute → render to a WebGPU canvas. Lives across all editing
// operations so slider drags only touch GPU (no JS↔wasm pixel readback).
//
// Pipeline:
//   1. Source RGBA (u32-packed bytes) lives in `source_buf`, uploaded once
//      per image via `gpu_set_source`.
//   2. The remap compute shader reads source_buf + uniform params, writes
//      vec4f remapped pixels into `remap_buf`. Slider drags hit this.
//   3. The carve compute shader reads remap_buf + the carve LUT, writes
//      rgba16float pixels into `output_tex`. The LUT is rebuilt CPU-side
//      whenever target dimensions change.
//   4. A trivial render pipeline (fullscreen triangle + textureSample)
//      samples output_tex into the WebGPU canvas current texture.
//
// The download path is the only consumer that reads pixels back to CPU.

const BUF_UNIFORM: u32 = 64;
const TEX_BINDING: u32 = 4;
const TEX_STORAGE: u32 = 8;

struct GpuCtx {
    device: JsValue,
    queue: JsValue,

    // Source buffer (u32-packed RGBA) and its dimensions.
    source_buf: JsValue,
    src_w: u32,
    src_h: u32,
    src_n: u32,

    // Remap compute pipeline + intermediate vec4f buffer.
    remap_pipeline: JsValue,
    remap_bind_group: JsValue,
    remap_buf: JsValue, // vec4f, src_n entries

    // Carve compute pipeline + LUT buffer + output storage texture.
    carve_pipeline: JsValue,
    carve_bind_group: Option<JsValue>,
    carve_dims_buf: JsValue, // uniform: target_w, target_h
    lut_buf: Option<JsValue>,
    output_tex: Option<JsValue>,
    output_tex_view: Option<JsValue>,
    target_w: u32,
    target_h: u32,

    // Render pipeline + canvas context.
    render_pipeline: JsValue,
    render_bgl: JsValue,
    render_bind_group: Option<JsValue>,
    sampler: JsValue,
    canvas_ctx: JsValue,
    canvas_format: String,

    // Shared params uniform (l/c min/max, hue cos/sin).
    params_buf: JsValue,

    // Optional readback buffer for the download path. Created lazily on
    // first download since downloads are infrequent.
    read_buf: RefCell<Option<JsValue>>,
}

thread_local! {
    static GPU_CTX: RefCell<Option<GpuCtx>> = const { RefCell::new(None) };
}

fn remap_shader() -> &'static str {
    r#"
struct Params {
  l_min: f32,
  l_max: f32,
  c_min: f32,
  c_max: f32,
  hue_cos: f32,
  hue_sin: f32,
  show_hdr: f32,
  time: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
}

@group(0) @binding(0) var<storage, read> rgba_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> linp3_out: array<vec4f>;
@group(0) @binding(2) var<uniform> params: Params;

fn srgb_to_linear(c: f32) -> f32 {
  if (c <= 0.04045) { return c / 12.92; }
  return pow((c + 0.055) / 1.055, 2.4);
}

// Cube root that preserves sign for negative inputs (which can occur for
// out-of-gamut LMS values when extrapolating boosts).
fn cbrt_signed(x: f32) -> f32 {
  return sign(x) * pow(abs(x), 1.0 / 3.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let n = arrayLength(&rgba_in);
  if (idx >= n) { return; }

  let p = rgba_in[idx];
  let r8 = f32(p & 0xFFu) / 255.0;
  let g8 = f32((p >> 8u) & 0xFFu) / 255.0;
  let b8 = f32((p >> 16u) & 0xFFu) / 255.0;
  let a8 = f32((p >> 24u) & 0xFFu) / 255.0;

  let lr = srgb_to_linear(r8);
  let lg = srgb_to_linear(g8);
  let lb = srgb_to_linear(b8);

  // Linear Display P3 → LMS (precomputed M_xyz_to_lms · M_p3_to_xyz).
  let lms_l = 0.4813371 * lr + 0.4620734 * lg + 0.0565038 * lb;
  let lms_m = 0.2288498 * lr + 0.6532486 * lg + 0.1179665 * lb;
  let lms_s = 0.0839833 * lr + 0.2242765 * lg + 0.6922382 * lb;

  // Nonlinear LMS' (cube root)
  let l_ = cbrt_signed(lms_l);
  let m_ = cbrt_signed(lms_m);
  let s_ = cbrt_signed(lms_s);

  // LMS' → OKLab, then scale to match the historical 0..100 / 0..160 ranges.
  var L = (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_) * 100.0;
  var A = (1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_) * 400.0;
  var B = (0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_) * 400.0;

  let l_scale = (params.l_max - params.l_min) / 100.0;
  L = params.l_min + L * l_scale;

  let chroma = sqrt(A * A + B * B);
  if (chroma > 0.5) {
    let chroma_norm = chroma / 160.0;
    let c_min_n = params.c_min / 100.0;
    let c_max_n = params.c_max / 100.0;
    let out_chroma_norm = c_min_n + (c_max_n - c_min_n) * chroma_norm;
    let factor = (out_chroma_norm * 160.0) / chroma;
    A = A * factor;
    B = B * factor;
  }

  // Hue rotation: rotate (A, B) by the precomputed cos/sin pair.
  let new_a = A * params.hue_cos - B * params.hue_sin;
  let new_b = A * params.hue_sin + B * params.hue_cos;
  A = new_a;
  B = new_b;

  // Unscale and invert OKLab.
  let big_l = L / 100.0;
  let aa = A / 400.0;
  let bb = B / 400.0;
  let l2_ = big_l + 0.3963377774 * aa + 0.2158037573 * bb;
  let m2_ = big_l - 0.1055613458 * aa - 0.0638541728 * bb;
  let s2_ = big_l - 0.0894841775 * aa - 1.2914855480 * bb;

  let lin_l = l2_ * l2_ * l2_;
  let lin_m = m2_ * m2_ * m2_;
  let lin_s = s2_ * s2_ * s2_;

  // LMS → linear Display P3 (precomputed M_xyz_to_p3 · M_lms_to_xyz).
  let lr2 =  3.1280366 * lin_l - 2.2571161 * lin_m + 0.1292834 * lin_s;
  let lg2 = -1.0911312 * lin_l + 2.4133403 * lin_m - 0.3221775 * lin_s;
  let lb2 = -0.0260093 * lin_l - 0.5080364 * lin_m + 1.5332833 * lin_s;

  // HDR view: any channel above 1.0 in linear extended P3 will clip on
  // SDR displays. Mark those pixels with a color that oscillates between
  // red and green based on `time` so they pop visually.
  if (params.show_hdr > 0.5) {
    let max_ch = max(max(lr2, lg2), lb2);
    if (max_ch > 1.0) {
      let t = 0.5 + 0.5 * sin(params.time * 4.0);
      linp3_out[idx] = vec4f(t, 1.0 - t, 0.0, a8);
      return;
    }
  }

  // Apply the Display P3 (= sRGB) transfer function. Chrome treats values
  // written to an rgba16float canvas as already encoded in the configured
  // color space, so we need to encode here. Sign-preserving for HDR-ready
  // extended-range values that may go negative or above 1.
  let er = sign(lr2) * select(1.055 * pow(abs(lr2), 1.0 / 2.4) - 0.055, 12.92 * abs(lr2), abs(lr2) <= 0.0031308);
  let eg = sign(lg2) * select(1.055 * pow(abs(lg2), 1.0 / 2.4) - 0.055, 12.92 * abs(lg2), abs(lg2) <= 0.0031308);
  let eb = sign(lb2) * select(1.055 * pow(abs(lb2), 1.0 / 2.4) - 0.055, 12.92 * abs(lb2), abs(lb2) <= 0.0031308);

  linp3_out[idx] = vec4f(er, eg, eb, a8);
}
"#
}

fn carve_shader() -> &'static str {
    r#"
@group(0) @binding(0) var<storage, read> linp3_in: array<vec4f>;
@group(0) @binding(1) var<storage, read> lut: array<u32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> dims: vec4u; // target_w, target_h, _, _

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tw = dims.x;
  let th = dims.y;
  if (gid.x >= tw || gid.y >= th) { return; }
  let out_idx = gid.y * tw + gid.x;
  let src_idx = lut[out_idx];
  let pixel = linp3_in[src_idx];
  textureStore(output_tex, vec2i(i32(gid.x), i32(gid.y)), pixel);
}
"#
}

fn render_shader() -> &'static str {
    r#"
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  // Fullscreen triangle (covers the viewport with 3 vertices).
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  var out: VsOut;
  out.pos = vec4f(pos[vid], 0.0, 1.0);
  // Map clip-space (-1..1) to uv (0..1) and flip y.
  out.uv = (pos[vid] + vec2f(1.0)) * 0.5;
  out.uv.y = 1.0 - out.uv.y;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
  return textureSample(src_tex, src_smp, in.uv);
}
"#
}

// ── Bind group layout helpers (specific to the new context) ─────────────────

fn make_remap_bgl(device: &JsValue) -> Result<JsValue, JsValue> {
    let entries = Array::new();
    entries.push(&js_obj(&[
        ("binding", 0u32.into()),
        ("visibility", SHADER_COMPUTE.into()),
        (
            "buffer",
            js_obj(&[("type", "read-only-storage".into())]).into(),
        ),
    ]));
    entries.push(&js_obj(&[
        ("binding", 1u32.into()),
        ("visibility", SHADER_COMPUTE.into()),
        ("buffer", js_obj(&[("type", "storage".into())]).into()),
    ]));
    entries.push(&js_obj(&[
        ("binding", 2u32.into()),
        ("visibility", SHADER_COMPUTE.into()),
        ("buffer", js_obj(&[("type", "uniform".into())]).into()),
    ]));
    js_call1(
        device,
        "createBindGroupLayout",
        &js_obj(&[("entries", entries.into())]),
    )
}

fn make_carve_bgl(device: &JsValue) -> Result<JsValue, JsValue> {
    let entries = Array::new();
    // 0: linp3_in (read-only storage)
    entries.push(&js_obj(&[
        ("binding", 0u32.into()),
        ("visibility", SHADER_COMPUTE.into()),
        (
            "buffer",
            js_obj(&[("type", "read-only-storage".into())]).into(),
        ),
    ]));
    // 1: lut (read-only storage)
    entries.push(&js_obj(&[
        ("binding", 1u32.into()),
        ("visibility", SHADER_COMPUTE.into()),
        (
            "buffer",
            js_obj(&[("type", "read-only-storage".into())]).into(),
        ),
    ]));
    // 2: output_tex (storage texture, write-only)
    entries.push(&js_obj(&[
        ("binding", 2u32.into()),
        ("visibility", SHADER_COMPUTE.into()),
        (
            "storageTexture",
            js_obj(&[
                ("access", "write-only".into()),
                ("format", "rgba16float".into()),
                ("viewDimension", "2d".into()),
            ])
            .into(),
        ),
    ]));
    // 3: dims uniform
    entries.push(&js_obj(&[
        ("binding", 3u32.into()),
        ("visibility", SHADER_COMPUTE.into()),
        ("buffer", js_obj(&[("type", "uniform".into())]).into()),
    ]));
    js_call1(
        device,
        "createBindGroupLayout",
        &js_obj(&[("entries", entries.into())]),
    )
}

const SHADER_VERTEX: u32 = 1;
const SHADER_FRAGMENT: u32 = 2;

fn make_render_bgl(device: &JsValue) -> Result<JsValue, JsValue> {
    let entries = Array::new();
    // 0: source texture (sampled)
    entries.push(&js_obj(&[
        ("binding", 0u32.into()),
        ("visibility", SHADER_FRAGMENT.into()),
        (
            "texture",
            js_obj(&[
                ("sampleType", "float".into()),
                ("viewDimension", "2d".into()),
            ])
            .into(),
        ),
    ]));
    // 1: sampler
    entries.push(&js_obj(&[
        ("binding", 1u32.into()),
        ("visibility", SHADER_FRAGMENT.into()),
        ("sampler", js_obj(&[("type", "filtering".into())]).into()),
    ]));
    js_call1(
        device,
        "createBindGroupLayout",
        &js_obj(&[("entries", entries.into())]),
    )
}

// ── Context creation ────────────────────────────────────────────────────────

async fn create_gpu_ctx(canvas: &JsValue) -> Result<GpuCtx, JsValue> {
    let nav = js_get(&js_sys::global(), "navigator")?;
    let gpu = js_get(&nav, "gpu")?;
    if gpu.is_undefined() || gpu.is_null() {
        return Err("WebGPU not available".into());
    }
    let adapter = js_await(js_call0(&gpu, "requestAdapter")?).await?;
    if adapter.is_null() || adapter.is_undefined() {
        return Err("No GPU adapter".into());
    }
    let max_storage = js_get(&js_get(&adapter, "limits")?, "maxStorageBufferBindingSize")?
        .as_f64()
        .unwrap_or(134217728.0);
    let required_limits = js_obj(&[(
        "maxStorageBufferBindingSize",
        JsValue::from_f64(max_storage),
    )]);
    let device_desc = js_obj(&[("requiredLimits", required_limits.into())]);
    let device = js_await(js_call1(&adapter, "requestDevice", &device_desc)?).await?;
    let queue = js_get(&device, "queue")?;

    // ── Configure the canvas WebGPU context ─────────────────────────────
    let canvas_ctx = js_call1(canvas, "getContext", &"webgpu".into())?;
    if canvas_ctx.is_null() || canvas_ctx.is_undefined() {
        return Err("canvas getContext('webgpu') returned null".into());
    }
    // rgba16float lets us carry extended-range Display P3 values for HDR.
    // toneMapping: 'extended' tells the compositor to use the display's
    // headroom for values above 1.0 on HDR displays; SDR displays clip.
    let canvas_format = "rgba16float".to_string();
    let tone_mapping = js_obj(&[("mode", "extended".into())]);
    let configure_desc = js_obj(&[
        ("device", device.clone()),
        ("format", canvas_format.as_str().into()),
        ("colorSpace", "display-p3".into()),
        ("toneMapping", tone_mapping.into()),
        ("alphaMode", "premultiplied".into()),
    ]);
    js_call1(&canvas_ctx, "configure", &configure_desc)?;

    // ── Remap pipeline ──────────────────────────────────────────────────
    let remap_module = create_shader(&device, remap_shader())?;
    let remap_bgl = make_remap_bgl(&device)?;
    let remap_pl_layout = {
        let layouts = Array::of1(&remap_bgl);
        js_call1(
            &device,
            "createPipelineLayout",
            &js_obj(&[("bindGroupLayouts", layouts.into())]),
        )?
    };
    let remap_pipeline = create_pipeline(&device, &remap_pl_layout, &remap_module, "main")?;

    // ── Carve pipeline ──────────────────────────────────────────────────
    let carve_module = create_shader(&device, carve_shader())?;
    let carve_bgl = make_carve_bgl(&device)?;
    let carve_pl_layout = {
        let layouts = Array::of1(&carve_bgl);
        js_call1(
            &device,
            "createPipelineLayout",
            &js_obj(&[("bindGroupLayouts", layouts.into())]),
        )?
    };
    let carve_pipeline = create_pipeline(&device, &carve_pl_layout, &carve_module, "main")?;

    // ── Render pipeline ─────────────────────────────────────────────────
    let render_module = create_shader(&device, render_shader())?;
    let render_bgl = make_render_bgl(&device)?;
    let render_pl_layout = {
        let layouts = Array::of1(&render_bgl);
        js_call1(
            &device,
            "createPipelineLayout",
            &js_obj(&[("bindGroupLayouts", layouts.into())]),
        )?
    };
    let render_pipeline = {
        let vertex = js_obj(&[
            ("module", render_module.clone()),
            ("entryPoint", "vs_main".into()),
        ]);
        let target = js_obj(&[("format", canvas_format.as_str().into())]);
        let targets = Array::of1(&target);
        let fragment = js_obj(&[
            ("module", render_module.clone()),
            ("entryPoint", "fs_main".into()),
            ("targets", targets.into()),
        ]);
        let primitive = js_obj(&[("topology", "triangle-list".into())]);
        let desc = js_obj(&[
            ("layout", render_pl_layout),
            ("vertex", vertex.into()),
            ("fragment", fragment.into()),
            ("primitive", primitive.into()),
        ]);
        js_call1(&device, "createRenderPipeline", &desc)?
    };

    // ── Sampler + persistent params/dims buffers ────────────────────────
    let sampler = js_call1(
        &device,
        "createSampler",
        &js_obj(&[
            ("magFilter", "linear".into()),
            ("minFilter", "linear".into()),
        ]),
    )?;

    // 12 f32 params (48 bytes) — uniform alignment
    let params_buf = create_buffer(&device, 48, BUF_UNIFORM | BUF_COPY_DST)?;
    // 4 u32 dims (16 bytes) for the carve shader
    let carve_dims_buf = create_buffer(&device, 16, BUF_UNIFORM | BUF_COPY_DST)?;

    // Source/remap buffers and textures are created lazily by gpu_set_source
    // and gpu_set_carve_lut once the dimensions are known.
    Ok(GpuCtx {
        device,
        queue,
        source_buf: JsValue::null(),
        src_w: 0,
        src_h: 0,
        src_n: 0,
        remap_pipeline,
        remap_bind_group: JsValue::null(),
        remap_buf: JsValue::null(),
        carve_pipeline,
        carve_bind_group: None,
        carve_dims_buf,
        lut_buf: None,
        output_tex: None,
        output_tex_view: None,
        target_w: 0,
        target_h: 0,
        render_pipeline,
        render_bgl,
        render_bind_group: None,
        sampler,
        canvas_ctx,
        canvas_format,
        params_buf,
        read_buf: RefCell::new(None),
    })
}

#[wasm_bindgen]
pub async fn gpu_init(canvas: JsValue) -> Result<(), JsValue> {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let ctx = create_gpu_ctx(&canvas).await?;
    GPU_CTX.with(|c| *c.borrow_mut() = Some(ctx));
    Ok(())
}

#[wasm_bindgen]
pub fn gpu_set_source(image_data: &[u8], w: u32, h: u32) -> Result<(), JsValue> {
    GPU_CTX.with(|c| -> Result<(), JsValue> {
        let mut c = c.borrow_mut();
        let ctx = c.as_mut().ok_or_else(|| JsValue::from("no gpu ctx"))?;
        let n = w * h;

        // (Re)allocate source + remap buffers if the dimensions changed.
        if ctx.src_w != w || ctx.src_h != h {
            if !ctx.source_buf.is_null() {
                let _ = js_call0(&ctx.source_buf, "destroy");
            }
            if !ctx.remap_buf.is_null() {
                let _ = js_call0(&ctx.remap_buf, "destroy");
            }
            ctx.source_buf = create_buffer(&ctx.device, n * 4, BUF_STORAGE | BUF_COPY_DST)?;
            // remap_buf holds vec4f per pixel = 16 bytes
            ctx.remap_buf = create_buffer(&ctx.device, n * 16, BUF_STORAGE)?;
            ctx.src_w = w;
            ctx.src_h = h;
            ctx.src_n = n;

            // Recreate the remap bind group against the new buffers.
            let bgl = make_remap_bgl(&ctx.device)?;
            ctx.remap_bind_group = create_bind_group(
                &ctx.device,
                &bgl,
                &[&ctx.source_buf, &ctx.remap_buf, &ctx.params_buf],
            )?;
        }

        write_buffer_u8(&ctx.queue, &ctx.source_buf, image_data)?;
        Ok(())
    })
}

#[wasm_bindgen]
pub fn gpu_set_carve_lut(lut: &[u32], target_w: u32, target_h: u32) -> Result<(), JsValue> {
    GPU_CTX.with(|c| -> Result<(), JsValue> {
        let mut c = c.borrow_mut();
        let ctx = c.as_mut().ok_or_else(|| JsValue::from("no gpu ctx"))?;
        let count = (target_w * target_h) as usize;
        if lut.len() != count {
            return Err(format!("lut len {} != target_w*target_h {count}", lut.len()).into());
        }

        // (Re)allocate the LUT buffer + output texture if dimensions changed.
        let dims_changed = ctx.target_w != target_w || ctx.target_h != target_h;
        if dims_changed {
            if let Some(old) = ctx.lut_buf.take() {
                let _ = js_call0(&old, "destroy");
            }
            if let Some(old) = ctx.output_tex.take() {
                let _ = js_call0(&old, "destroy");
            }
            ctx.output_tex_view = None;

            let lut_buf =
                create_buffer(&ctx.device, (count * 4) as u32, BUF_STORAGE | BUF_COPY_DST)?;
            ctx.lut_buf = Some(lut_buf);

            // Storage texture for the carved+remapped result.
            let size = js_obj(&[
                ("width", target_w.into()),
                ("height", target_h.into()),
                ("depthOrArrayLayers", 1u32.into()),
            ]);
            let tex_desc = js_obj(&[
                ("size", size.into()),
                ("format", "rgba16float".into()),
                ("usage", (TEX_BINDING | TEX_STORAGE | 1 /* COPY_SRC */).into()),
            ]);
            let tex = js_call1(&ctx.device, "createTexture", &tex_desc)?;
            let view = js_call1(&tex, "createView", &Object::new())?;
            ctx.output_tex = Some(tex);
            ctx.output_tex_view = Some(view);

            ctx.target_w = target_w;
            ctx.target_h = target_h;

            // Recreate carve + render bind groups against the new resources.
            let carve_bgl = make_carve_bgl(&ctx.device)?;
            ctx.carve_bind_group = Some(make_carve_bind_group(
                &ctx.device,
                &carve_bgl,
                ctx.remap_buf.clone(),
                ctx.lut_buf.as_ref().unwrap().clone(),
                ctx.output_tex_view.as_ref().unwrap().clone(),
                ctx.carve_dims_buf.clone(),
            )?);

            ctx.render_bind_group = Some(make_render_bind_group(
                &ctx.device,
                &ctx.render_bgl,
                ctx.output_tex_view.as_ref().unwrap().clone(),
                ctx.sampler.clone(),
            )?);

            // Update the carve dims uniform.
            let dims = [target_w, target_h, 0u32, 0u32];
            let dims_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(dims.as_ptr() as *const u8, 16) };
            write_buffer_u8(&ctx.queue, &ctx.carve_dims_buf, dims_bytes)?;
        }

        write_buffer_u32(&ctx.queue, ctx.lut_buf.as_ref().unwrap(), lut)?;
        Ok(())
    })
}

fn make_carve_bind_group(
    device: &JsValue,
    layout: &JsValue,
    linp3: JsValue,
    lut: JsValue,
    tex_view: JsValue,
    dims_buf: JsValue,
) -> Result<JsValue, JsValue> {
    let entries = Array::new();
    entries.push(&js_obj(&[
        ("binding", 0u32.into()),
        ("resource", js_obj(&[("buffer", linp3)]).into()),
    ]));
    entries.push(&js_obj(&[
        ("binding", 1u32.into()),
        ("resource", js_obj(&[("buffer", lut)]).into()),
    ]));
    entries.push(&js_obj(&[("binding", 2u32.into()), ("resource", tex_view)]));
    entries.push(&js_obj(&[
        ("binding", 3u32.into()),
        ("resource", js_obj(&[("buffer", dims_buf)]).into()),
    ]));
    js_call1(
        device,
        "createBindGroup",
        &js_obj(&[("layout", layout.clone()), ("entries", entries.into())]),
    )
}

fn make_render_bind_group(
    device: &JsValue,
    layout: &JsValue,
    tex_view: JsValue,
    sampler: JsValue,
) -> Result<JsValue, JsValue> {
    let entries = Array::new();
    entries.push(&js_obj(&[("binding", 0u32.into()), ("resource", tex_view)]));
    entries.push(&js_obj(&[("binding", 1u32.into()), ("resource", sampler)]));
    js_call1(
        device,
        "createBindGroup",
        &js_obj(&[("layout", layout.clone()), ("entries", entries.into())]),
    )
}

#[wasm_bindgen]
pub fn gpu_render(
    l_min: f32,
    l_max: f32,
    c_min: f32,
    c_max: f32,
    hue_deg: f32,
    show_hdr: f32,
    time: f32,
) -> Result<(), JsValue> {
    GPU_CTX.with(|c| -> Result<(), JsValue> {
        let c = c.borrow();
        let ctx = c.as_ref().ok_or_else(|| JsValue::from("no gpu ctx"))?;
        if ctx.lut_buf.is_none() || ctx.output_tex_view.is_none() {
            return Err("gpu_render called before gpu_set_carve_lut".into());
        }

        // Upload params.
        let hue_rad = hue_deg.to_radians();
        let params = [
            l_min,
            l_max,
            c_min,
            c_max,
            hue_rad.cos(),
            hue_rad.sin(),
            show_hdr,
            time,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let params_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, 48) };
        write_buffer_u8(&ctx.queue, &ctx.params_buf, params_bytes)?;

        // Encode all three passes into one command buffer.
        let enc = js_call1(&ctx.device, "createCommandEncoder", &Object::new())?;

        // Pass 1: remap compute → remap_buf
        dispatch(
            &enc,
            &ctx.remap_pipeline,
            &ctx.remap_bind_group,
            ctx.src_n.div_ceil(256),
        )?;

        // Pass 2: carve compute → output_tex
        let carve_x = ctx.target_w.div_ceil(8);
        let carve_y = ctx.target_h.div_ceil(8);
        let carve_bg = ctx.carve_bind_group.as_ref().unwrap();
        let pass = js_call1(&enc, "beginComputePass", &Object::new())?;
        js_call1(&pass, "setPipeline", &ctx.carve_pipeline)?;
        js_call2(&pass, "setBindGroup", &0.into(), carve_bg)?;
        let dispatch_fn: Function = js_get(&pass, "dispatchWorkgroups")?.dyn_into()?;
        Reflect::apply(
            &dispatch_fn,
            &pass,
            &Array::of2(&carve_x.into(), &carve_y.into()),
        )?;
        js_call0(&pass, "end")?;

        // Pass 3: render output_tex → canvas
        let canvas_tex = js_call0(&ctx.canvas_ctx, "getCurrentTexture")?;
        let canvas_view = js_call1(&canvas_tex, "createView", &Object::new())?;
        let color_attachment = js_obj(&[
            ("view", canvas_view),
            ("loadOp", "clear".into()),
            ("storeOp", "store".into()),
            (
                "clearValue",
                js_obj(&[
                    ("r", 0.0.into()),
                    ("g", 0.0.into()),
                    ("b", 0.0.into()),
                    ("a", 1.0.into()),
                ])
                .into(),
            ),
        ]);
        let attachments = Array::of1(&color_attachment);
        let render_desc = js_obj(&[("colorAttachments", attachments.into())]);
        let rpass = js_call1(&enc, "beginRenderPass", &render_desc)?;
        js_call1(&rpass, "setPipeline", &ctx.render_pipeline)?;
        let render_bg = ctx.render_bind_group.as_ref().unwrap();
        js_call2(&rpass, "setBindGroup", &0.into(), render_bg)?;
        let draw_fn: Function = js_get(&rpass, "draw")?.dyn_into()?;
        draw_fn.call1(&rpass, &3.into())?; // 3 vertices, fullscreen triangle
        js_call0(&rpass, "end")?;

        let cmd = js_call0(&enc, "finish")?;
        js_call1(&ctx.queue, "submit", &Array::of1(&cmd))?;
        Ok(())
    })
}

// Convert IEEE 754 half-precision (f16) bits to f32. Subnormals flush to
// zero — fine for our use since they clamp to 0 in the SDR encode anyway.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x3FF) as u32;
    let bits: u32 = if exp == 0 {
        sign << 31
    } else if exp == 31 {
        (sign << 31) | (0xFFu32 << 23) | (mant << 13)
    } else {
        (sign << 31) | (((exp + 112) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[wasm_bindgen]
pub async fn gpu_readback_rgba8() -> Result<Uint8Array, JsValue> {
    // Snapshot the resources we need so we don't hold a RefCell borrow
    // across an await point.
    let (device, queue, output_tex, target_w, target_h) =
        GPU_CTX.with(|c| -> Result<_, JsValue> {
            let c = c.borrow();
            let ctx = c.as_ref().ok_or_else(|| JsValue::from("no gpu ctx"))?;
            let tex = ctx
                .output_tex
                .as_ref()
                .ok_or_else(|| JsValue::from("no output texture"))?;
            Ok((
                ctx.device.clone(),
                ctx.queue.clone(),
                tex.clone(),
                ctx.target_w,
                ctx.target_h,
            ))
        })?;

    let bpp = 8u32; // rgba16float = 8 bytes/pixel
    let bytes_per_row_unpadded = target_w * bpp;
    // copyTextureToBuffer requires bytesPerRow to be a multiple of 256.
    let bytes_per_row = bytes_per_row_unpadded.div_ceil(256) * 256;
    let buf_size = bytes_per_row * target_h;

    let staging = create_buffer(&device, buf_size, BUF_MAP_READ | BUF_COPY_DST)?;

    let enc = js_call1(&device, "createCommandEncoder", &Object::new())?;
    let src = js_obj(&[("texture", output_tex)]);
    let dst = js_obj(&[
        ("buffer", staging.clone()),
        ("bytesPerRow", bytes_per_row.into()),
        ("rowsPerImage", target_h.into()),
    ]);
    let extent = js_obj(&[
        ("width", target_w.into()),
        ("height", target_h.into()),
        ("depthOrArrayLayers", 1u32.into()),
    ]);
    let copy_fn: Function = js_get(&enc, "copyTextureToBuffer")?.dyn_into()?;
    Reflect::apply(
        &copy_fn,
        &enc,
        &Array::of3(&src.into(), &dst.into(), &extent.into()),
    )?;

    let cmd = js_call0(&enc, "finish")?;
    js_call1(&queue, "submit", &Array::of1(&cmd))?;

    // Map for read. mapAsync(GPUMapMode.READ = 1) → Promise<undefined>.
    let map_promise = js_call1(&staging, "mapAsync", &1u32.into())?;
    js_await(map_promise).await?;

    let mapped = js_call0(&staging, "getMappedRange")?;
    let raw_view = Uint8Array::new(&mapped);
    let mut raw = vec![0u8; buf_size as usize];
    raw_view.copy_to(&mut raw[..]);

    let mut out = vec![0u8; (target_w * target_h * 4) as usize];
    for y in 0..target_h {
        let row_start = (y * bytes_per_row) as usize;
        for x in 0..target_w {
            for ch in 0..4u32 {
                let off = row_start + ((x * bpp) + ch * 2) as usize;
                let h = (raw[off] as u16) | ((raw[off + 1] as u16) << 8);
                let v = f16_to_f32(h).clamp(0.0, 1.0);
                let out_idx = ((y * target_w + x) * 4 + ch) as usize;
                out[out_idx] = (v * 255.0 + 0.5) as u8;
            }
        }
    }

    let _ = js_call0(&staging, "unmap");
    let _ = js_call0(&staging, "destroy");

    Ok(Uint8Array::from(&out[..]))
}

// Inverse sRGB / Display P3 transfer (decode encoded → linear).
// Sign-preserving so extended-range HDR values keep their sign.
fn srgb_decode(v: f32) -> f32 {
    let a = v.abs();
    let lin = if a <= 0.04045 {
        a / 12.92
    } else {
        ((a + 0.055) / 1.055).powf(2.4)
    };
    lin.copysign(v)
}

// SMPTE ST 2084 (PQ) inverse EOTF: linear normalized luminance [0,1]
// (where 1.0 = 10000 nits) → encoded signal [0,1].
fn pq_encode(y: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let y = y.max(0.0);
    let yp = y.powf(M1);
    ((C1 + C2 * yp) / (1.0 + C3 * yp)).powf(M2)
}

// Read back the output texture as PQ-encoded BT.2020 16-bit RGB. Used for
// the HDR AVIF export path. The texture currently stores sRGB-encoded
// extended-range Display P3 values; we undo the sRGB transfer, convert
// linear D65 P3 → linear D65 BT.2020 (matrix from BT.2407), apply the
// PQ EOTF inverse using a 203-nit HDR/SDR reference white per ITU BT.2408,
// and quantize into the bottom `depth` bits of a u16 (libavif convention).
#[wasm_bindgen]
pub async fn gpu_readback_hdr_pq_u16(depth: u32) -> Result<Uint16Array, JsValue> {
    let (device, queue, output_tex, target_w, target_h) =
        GPU_CTX.with(|c| -> Result<_, JsValue> {
            let c = c.borrow();
            let ctx = c.as_ref().ok_or_else(|| JsValue::from("no gpu ctx"))?;
            let tex = ctx
                .output_tex
                .as_ref()
                .ok_or_else(|| JsValue::from("no output texture"))?;
            Ok((
                ctx.device.clone(),
                ctx.queue.clone(),
                tex.clone(),
                ctx.target_w,
                ctx.target_h,
            ))
        })?;

    let bpp = 8u32; // rgba16float source
    let bytes_per_row_unpadded = target_w * bpp;
    let bytes_per_row = bytes_per_row_unpadded.div_ceil(256) * 256;
    let buf_size = bytes_per_row * target_h;

    let staging = create_buffer(&device, buf_size, BUF_MAP_READ | BUF_COPY_DST)?;

    let enc = js_call1(&device, "createCommandEncoder", &Object::new())?;
    let src = js_obj(&[("texture", output_tex)]);
    let dst = js_obj(&[
        ("buffer", staging.clone()),
        ("bytesPerRow", bytes_per_row.into()),
        ("rowsPerImage", target_h.into()),
    ]);
    let extent = js_obj(&[
        ("width", target_w.into()),
        ("height", target_h.into()),
        ("depthOrArrayLayers", 1u32.into()),
    ]);
    let copy_fn: Function = js_get(&enc, "copyTextureToBuffer")?.dyn_into()?;
    Reflect::apply(
        &copy_fn,
        &enc,
        &Array::of3(&src.into(), &dst.into(), &extent.into()),
    )?;
    let cmd = js_call0(&enc, "finish")?;
    js_call1(&queue, "submit", &Array::of1(&cmd))?;

    let map_promise = js_call1(&staging, "mapAsync", &1u32.into())?;
    js_await(map_promise).await?;

    let mapped = js_call0(&staging, "getMappedRange")?;
    let raw_view = Uint8Array::new(&mapped);
    let mut raw = vec![0u8; buf_size as usize];
    raw_view.copy_to(&mut raw[..]);

    let max_val: f32 = ((1u32 << depth) - 1) as f32;
    // 1.0 in linear extended P3 = SDR diffuse white. ITU BT.2408 maps SDR
    // diffuse white to 203 nits in HDR signals; PQ normalizes 10000 nits
    // to 1.0, so the scale is 203/10000.
    const SDR_WHITE_TO_PQ: f32 = 203.0 / 10000.0;

    let n = (target_w * target_h) as usize;
    let mut out = vec![0u16; n * 4];
    for y in 0..target_h {
        let row_start = (y * bytes_per_row) as usize;
        for x in 0..target_w {
            let pix_off = row_start + (x * bpp) as usize;
            let mut ch = [0f32; 4];
            for i in 0..4 {
                let off = pix_off + i * 2;
                let h = (raw[off] as u16) | ((raw[off + 1] as u16) << 8);
                ch[i] = f16_to_f32(h);
            }
            // Undo sRGB encoding → linear extended-range P3.
            let lr = srgb_decode(ch[0]);
            let lg = srgb_decode(ch[1]);
            let lb = srgb_decode(ch[2]);

            // Linear D65 P3 → linear D65 BT.2020 (matrix derived from
            // M_p3_to_xyz · M_xyz_to_2020 with a Bradford D65↔D65 noop).
            let r2020 = 0.7538330 * lr + 0.1985801 * lg + 0.0475869 * lb;
            let g2020 = 0.0457345 * lr + 0.9417696 * lg + 0.0124960 * lb;
            let b2020 = -0.0011422 * lr + 0.0176061 * lg + 0.9835361 * lb;

            // Scale into PQ-normalized luminance, encode with PQ inverse EOTF.
            let pr = pq_encode(r2020 * SDR_WHITE_TO_PQ).clamp(0.0, 1.0);
            let pg = pq_encode(g2020 * SDR_WHITE_TO_PQ).clamp(0.0, 1.0);
            let pb = pq_encode(b2020 * SDR_WHITE_TO_PQ).clamp(0.0, 1.0);

            let idx = ((y * target_w + x) as usize) * 4;
            out[idx] = (pr * max_val + 0.5) as u16;
            out[idx + 1] = (pg * max_val + 0.5) as u16;
            out[idx + 2] = (pb * max_val + 0.5) as u16;
            // Alpha pass-through (linear, no PQ) — quantized to the same depth.
            out[idx + 3] = (ch[3].clamp(0.0, 1.0) * max_val + 0.5) as u16;
        }
    }

    let _ = js_call0(&staging, "unmap");
    let _ = js_call0(&staging, "destroy");

    let arr = Uint16Array::new_with_length(out.len() as u32);
    arr.copy_from(&out);
    Ok(arr)
}

#[wasm_bindgen]
pub fn gpu_dispose() {
    GPU_CTX.with(|c| {
        c.borrow_mut().take();
    });
}
