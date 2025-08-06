struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat3x3<f32>,
}

struct CounterUniform {
    count: u32,
    _padding_1: u32,
    _padding_2: u32,
    _padding_3: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> counter: CounterUniform;

@group(2) @binding(0)
var<uniform> transform: TransformUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) face_normal: vec4<f32>,
    @builtin(vertex_index) face_index: u32,
};

// The output we send to our fragment shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) face_index: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    // We define the output we want to send over to frag shader
    var out: VertexOutput;
    let model_matrix = camera.view * transform.model;

    out.face_index = counter.count + model.face_index / 3;

    // We set the \"position\" by using the `clip_position` property
    // We multiply it by the camera position matrix and the instance position matrix
    out.clip_position = camera.proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
//u32 {
    //return bitcast<vec4<f32>>(res);
    // webgl dosen't support rendering to u32, so we have to resort to this
    let res = in.face_index;
    let f1 = f32((res >> u32(24))) / 255.;
    let f2 = f32(((res << u32(8)) >> u32(24))) / 255.;
    let f3 = f32(((res << u32(16)) >> u32(24))) / 255.;
    let f4 = f32(((res << u32(24)) >> u32(24))) / 255.;
    return vec4<f32>(f4, f3, f2, f1);
    //return unpack4x8unorm(in.face_index);
}
