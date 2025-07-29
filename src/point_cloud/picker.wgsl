struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}

struct CounterUniform {
    count: u32,
    _padding_1: u32,
    _padding_2: u32,
    _padding_3: u32,
}

struct SettingsUniform {
    radius: f32,
    char_len: f32,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> counter: CounterUniform;

@group(2) @binding(0)
var<uniform> transform: TransformUniform;
@group(3) @binding(0)
var<uniform> settings: SettingsUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct DataInput {
    @location(1) position: vec3<f32>,
    @builtin(instance_index) index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) index: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
    data: DataInput,
) -> VertexOutput {
    let model_matrix = transform.model;

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let camera_right = normalize(vec3<f32>(camera.view_proj[0].x, camera.view_proj[1].x, camera.view_proj[2].x));
    let camera_up = normalize(vec3<f32>(camera.view_proj[0].y, camera.view_proj[1].y, camera.view_proj[2].y));
    let center = (model_matrix * vec4<f32>(data.position, 1.)).xyz;
    let world_position = (model_matrix * vec4<f32>(data.position + (model.position.x * camera_right + model.position.y * camera_up) * settings.radius * settings.char_len, 1.)).xyz;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_pos = world_position;
    out.center = center;
    out.index = data.index;
    return out;
}

// function from :
// https://iquilezles.org/articles/intersectors/
fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }


fn sphIntersect(ro: vec3<f32>, rd: vec3<f32>, ce: vec3<f32>, ra: f32) -> vec2<f32> {
    let oc = ro - ce;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - ra * ra;
    var h = b * b - c;
    if h < 0.0 { return vec2<f32>(-1.0); } // no intersection
    h = sqrt(h);
    return vec2<f32>(-b - h, -b + h);
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let ro = camera.view_pos.xyz;
    let rd = normalize(in.world_pos - camera.view_pos.xyz);
    let ce = in.center;
    let det = determinant(transform.normal);
    let r = settings.radius * settings.char_len / pow(det, 1. / 3.);

    var out: FragOutput;

    let t = sphIntersect(ro, rd, ce, r);
    if t.x < 0.0 {
        discard;
    }
    let pos = ro + t.x * rd;

    let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
    out.depth = clip_space_pos.z / clip_space_pos.w;
    let res = counter.count + in.index;
    // webgl dosen't support rendering to u32, so we have to resort to this
    let f1 = f32((res >> u32(24))) / 255.;
    let f2 = f32(((res << u32(8)) >> u32(24))) / 255.;
    let f3 = f32(((res << u32(16)) >> u32(24))) / 255.;
    let f4 = f32(((res << u32(24)) >> u32(24))) / 255.;
    out.color = vec4<f32>(f4, f3, f2, f1);
    //return bitcast<vec4<f32>>(res);
    return out;
}
