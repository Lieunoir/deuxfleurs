use super::geometry::PointCloudData;
use crate::shader;

macro_rules! SHADER { () => {"
struct CameraUniform {{
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}}

struct TransformUniform {{
    model: mat4x4<f32>,
    normal: mat3x3<f32>,
    scale: f32,
}}

struct Jitter {{
    jitter: vec4<f32>,
}}

struct SettingsUniform {{
    radius: f32,
    char_len: f32,
    color: vec3<f32>,
}}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> jitter: Jitter;

@group(1) @binding(0)
var<uniform> transform: TransformUniform;
@group(2) @binding(0)
var<uniform> settings: SettingsUniform;
struct VertexInput {{
    @location(0) position: vec3<f32>,
}};

struct PosInput {{
    @location(1) position: vec3<f32>,
}};

// Data Input
{}

// Uniforms

{}

struct VertexOutput {{
    @builtin(position) clip_position: vec4<f32>,
	@location(0) view_pos: vec3<f32>,
	@location(1) center: vec3<f32>,
    {}
}};

@vertex
fn vs_main(
    model: VertexInput,
    pos: PosInput,
    {}
) -> VertexOutput {{
    let model_matrix = camera.view * transform.model;
    var out: VertexOutput;

    let camera_right = vec3<f32>(1., 0., 0.);
    let camera_up = vec3<f32>(0., 1., 0.);
    let center = (model_matrix * vec4<f32>(pos.position, 1.)).xyz;
    let view_position = center + (model.position.x * camera_right + model.position.y * camera_up) * settings.radius * settings.char_len * transform.scale;
    let clip_pos = camera.proj * vec4<f32>(view_position, 1.0);
    out.clip_position = clip_pos + jitter.jitter * clip_pos.w;
    out.view_pos = view_position;
    out.center = center;
    // Set output
    {}
    return out;
}}

fn sphIntersect( ro: vec3<f32>, rd: vec3<f32>, ce: vec3<f32>, ra: f32 ) -> vec2<f32>
{{
    let oc = ro - ce;
    let b = dot( oc, rd );
    let c = dot( oc, oc ) - ra*ra;
    var h = b*b - c;
    if( h<0.0 ) {{ return vec2<f32>(-1.0); }} // no intersection
    h = sqrt( h );
    return vec2<f32>( -b-h, -b+h );
}}

struct FragOutput {{
    @builtin(frag_depth) depth: f32,
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
}}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {{
    let ro = vec3<f32>(0.);
	let rd = normalize(in.view_pos);
    let ce = in.center;
    let r = settings.radius * settings.char_len * transform.scale;
    //let pa = in.orig_position;
    //let pb1 = in.orig_position + 0.5 * in.arrow * 0.1;
    //let pb2 = in.orig_position + in.arrow * 0.1;

    var out: FragOutput;

    let t = sphIntersect( ro, rd, ce, r);
    if(t.x < 0.0) {{
        discard;
    }}
	let pos = ro + t.x * rd;
	let normal = normalize(pos - ce);

    {}

    let clip_space_pos = camera.proj * vec4<f32>(pos, 1.);
	out.albedo = vec4<f32>(lambertian, 0.3);
    out.normal = vec4<f32>((normal + vec3<f32>(1.)) / 2. , 0.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}}
"};}

pub fn get_shader(data_format: Option<&PointCloudData>) -> String {
    let data_struct = match data_format {
        Some(PointCloudData::Scalar(..)) => {
            "
struct DataInput {
    @location(2) val: f32,
}"
        }
        Some(PointCloudData::Color(_)) => {
            "
struct DataInput {
    @location(2) val: vec3<f32>,
}"
        }
        None => "",
    };

    let uniform = match data_format {
        Some(PointCloudData::Scalar(..)) => shader::COLORMAP_UNIFORM,
        _ => "",
    };

    let output_val = match data_format {
        Some(PointCloudData::Scalar(..)) => "@location(3) val: f32,",
        Some(PointCloudData::Color(_)) => "@location(3) val: vec3<f32>",
        None => "",
    };

    let input_val = match data_format {
        Some(_) => "data: DataInput,",
        None => "",
    };

    let set_output = match data_format {
        Some(_) => "out.val = data.val;",
        None => "",
    };

    let color_output = match data_format {
        Some(PointCloudData::Scalar(..)) => "let lambertian = colormap(in.val);",
        Some(PointCloudData::Color(_)) => "let lambertian = in.val;",
        None => "let lambertian = settings.color;",
    };

    format!(
        SHADER!(),
        data_struct, uniform, output_val, input_val, set_output, color_output,
    )
}
