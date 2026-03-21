use js_sys::{Array, Function, Object, Promise, Reflect, Uint32Array, Uint8Array};
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
    js_get(obj, method)?.dyn_into::<Function>()?.call2(obj, a, b)
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
    js_call1(device, "createShaderModule", &js_obj(&[("code", code.into())]))
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

fn create_storage_bgl(device: &JsValue, count: u32, read_only_mask: u32) -> Result<JsValue, JsValue> {
    let entries = Array::new();
    for i in 0..count {
        let is_ro = (read_only_mask >> i) & 1 == 1;
        let buf_desc = js_obj(&[("type", if is_ro { "read-only-storage" } else { "storage" }.into())]);
        entries.push(&js_obj(&[
            ("binding", i.into()),
            ("visibility", SHADER_COMPUTE.into()),
            ("buffer", buf_desc.into()),
        ]));
    }
    js_call1(device, "createBindGroupLayout", &js_obj(&[("entries", entries.into())]))
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

fn lab_f(t: f32) -> f32 {{
  if (t > 216.0 / 24389.0) {{ return pow(t, 1.0 / 3.0); }}
  return (24389.0 / 27.0 * t + 16.0) / 116.0;
}}

@compute @workgroup_size(256)
fn convert(@builtin(global_invocation_id) gid: vec3u) {{
  let idx = gid.x;
  if (idx >= {n}u) {{ return; }}
  let p = rgba[idx];
  let lr = srgb_to_linear(f32(p & 0xFFu) / 255.0);
  let lg = srgb_to_linear(f32((p >> 8u) & 0xFFu) / 255.0);
  let lb = srgb_to_linear(f32((p >> 16u) & 0xFFu) / 255.0);
  let x = 0.4124564*lr + 0.3575761*lg + 0.1804375*lb;
  let y = 0.2126729*lr + 0.7151522*lg + 0.0721750*lb;
  let z = 0.0193339*lr + 0.1191920*lg + 0.9503041*lb;
  let fx = lab_f(x / 0.95047); let fy = lab_f(y); let fz = lab_f(z / 1.08883);
  lab_out[idx] = vec4f(116.0*fy - 16.0, 500.0*(fx - fy), 200.0*(fy - fz), 0.0);
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
    let i=row0+k; var active=false; var j=0u;
    if (k>0u && center>=k && jl<2u*k) {{
      j=center-k+jl; active=(i<MH)&&(i>0u)&&(j<w);
    }}
    if (active) {{ dp_step(i,j,w); }}
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
    let n_down = (w + B - 1) / B;
    let n_up = if n_down > 0 { n_down - 1 } else { 0 };
    let n_strips = (h + S - 1) / S;

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
    let device = js_await(js_call0(&adapter, "requestDevice")?).await?;
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
    dispatch(&enc, &lab_pipe, &lab_bg, (n + 255) / 256)?;
    let cmd_buf = js_call0(&enc, "finish")?;
    let cmds = Array::of1(&cmd_buf);
    js_call1(&queue, "submit", &cmds)?;

    // ── Seam carving loop ───────────────────────────────────────────────
    let seam_bg = create_bind_group(
        &device,
        &seam_bgl,
        &[&lab_b, &dp_b, &dirs_b, &seam_b, &cm_b, &order_b, &params_b],
    )?;

    let compact_wg = (h + 255) / 256;
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
            let args =
                Array::of5(&order_b, &0.into(), &read_b, &0.into(), &(n * 4).into());
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
