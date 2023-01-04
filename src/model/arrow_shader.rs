pub const ARROW_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VectorInput {
    @location(1) color: vec3<f32>,
    @location(2) orig_position: vec3<f32>,
    @location(3) arrow: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    vector_i: VectorInput,
) -> VertexOutput {
    //let model_matrix = transform.model;
    //let normal_matrix = transform.normal;

    // We define the output we want to send over to frag shader
    var out: VertexOutput;

    out.color = vector_i.color;
    let view_axis = normalize(vector_i.orig_position - camera.view_pos.xyz);
    let arrow_axis = normalize(vector_i.arrow);
    let right_axis = normalize(cross(view_axis, arrow_axis));
    let rotation_mat = mat3x3<f32>(
        right_axis,
        arrow_axis,
        view_axis);
    let position = rotation_mat * (model.position - vec3<f32>(0.5, 0., 0.)) * 0.1 + vec3<f32>(vector_i.orig_position);

    //out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let arrow_color = vec4<f32>(in.color, 1.);
    let result = arrow_color;

    return result;
}
";
