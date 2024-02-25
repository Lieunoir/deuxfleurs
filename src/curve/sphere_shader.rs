use super::CurveData;
use crate::shader;

macro_rules! SHADER { () => {"
struct CameraUniform {{
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}}

struct Light {{
    position: vec3<f32>,
    color: vec3<f32>,
}}

struct TransformUniform {{
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}}

struct Jitter {{
    jitter: vec4<f32>,
}}

struct SettingsUniform {{
    radius: f32,
    color: vec3<f32>,
}}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;
@group(0) @binding(2)
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
	@location(0) world_pos: vec3<f32>,
	@location(1) center: vec3<f32>,
    {}
}};

@vertex
fn vs_main(
    model: VertexInput,
    pos: PosInput,
    {}
) -> VertexOutput {{
    let model_matrix = transform.model;

    //let world_vector_pos = (model_matrix * vec4<f32>(vector_i.orig_position, 1.)).xyz;
    //// Do we want to scale a vector field if we scale its attached mesh?
    //let world_vector_arrow_t = (model_matrix * vec4<f32>(vector_i.orig_position + vector_i.arrow, 1.)).xyz - world_vector_pos;
    //let arrow_ampl = length(world_vector_arrow_t);
    //let world_vector_arrow = normalize(world_vector_arrow_t);

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let camera_right = normalize(vec3<f32>(camera.view_proj.x.x, camera.view_proj.y.x, camera.view_proj.z.x));
    let camera_up = normalize(vec3<f32>(camera.view_proj.x.y, camera.view_proj.y.y, camera.view_proj.z.y));
    let world_position = (model_matrix * vec4<f32>(pos.position + (model.position.x * camera_right + model.position.y * camera_up) * settings.radius, 1.)).xyz;
    let clip_pos = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.clip_position = clip_pos + jitter.jitter * clip_pos.w;
    out.world_pos = world_position;
    out.center = (model_matrix * vec4<f32>(pos.position, 1.)).xyz;
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
    @location(0) position: vec4<f32>,
    @location(1) albedo: vec4<f32>,
    @location(2) normal: vec4<f32>,
}}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {{
    let ro = camera.view_pos.xyz;
	let rd = normalize(in.world_pos - camera.view_pos.xyz);
    let ce = in.center;
    let det = determinant(transform.normal);
    let r = settings.radius / pow(det, 1. / 3.);
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

    let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.albedo = vec4<f32>(lambertian, 1.);
    out.position = vec4<f32>(pos, 0.);
    out.normal = vec4<f32>((normal + vec3<f32>(1.)) / 2. , 0.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}}
"};}

pub fn get_shader(data_format: Option<&CurveData>) -> String {
    let data_struct = match data_format {
        Some(CurveData::Scalar(..)) => {
            "
struct DataInput {
    @location(2) val: f32,
}"
        }
        Some(CurveData::Color(_)) => {
            "
struct DataInput {
    @location(2) val: vec3<f32>,
}"
        }
        None => "",
    };

    let uniform = match data_format {
        Some(CurveData::Scalar(..)) => shader::COLORMAP_UNIFORM,
        _ => "",
    };

    let output_val = match data_format {
        Some(CurveData::Scalar(..)) => "@location(3) val: f32,",
        Some(CurveData::Color(_)) => "@location(3) val: vec3<f32>",
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
        Some(CurveData::Scalar(..)) => "let lambertian = colormap(in.val);",
        Some(CurveData::Color(_)) => "let lambertian = in.val;",
        None => "let lambertian = settings.color;",
    };

    format!(
        SHADER!(),
        data_struct,
        uniform,
        output_val,
        input_val,
        set_output,
        color_output,
    )
}
