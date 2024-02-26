pub const ARROW_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}

struct SettingsUniform {
    magnitude: f32,
    color: vec3<f32>,
}

struct Jitter {
    jitter: vec4<f32>,
}

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
struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VectorInput {
    @location(1) orig_position: vec3<f32>,
    @location(2) arrow: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) world_pos: vec3<f32>,
	@location(1) orig_position: vec3<f32>,
	@location(2) arrow: vec3<f32>,
	@location(3) radius: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
    vector_i: VectorInput,
) -> VertexOutput {
    let model_matrix = transform.model;
    let normal_matrix = transform.normal;

    let world_vector_pos = (model_matrix * vec4<f32>(vector_i.orig_position, 1.)).xyz;
    // Do we want to scale a vector field if we scale its attached mesh?
    let world_vector_arrow_t = (model_matrix * vec4<f32>(vector_i.orig_position + vector_i.arrow, 1.)).xyz - world_vector_pos;
    let arrow_ampl = length(world_vector_arrow_t);
    let world_vector_arrow = normalize(world_vector_arrow_t);

    // We define the output we want to send over to frag shader
    var out: VertexOutput;

	out.orig_position = world_vector_pos;
	out.arrow = world_vector_arrow * settings.magnitude * arrow_ampl;

    let view_axis = normalize(world_vector_pos - camera.view_pos.xyz);
    let arrow_axis = world_vector_arrow;
    let right_axis = (model_matrix * vec4<f32>(normalize(cross(view_axis, arrow_axis)), 0.)).xyz;
    let depth_axis = (model_matrix * vec4<f32>(-normalize(cross(arrow_axis, right_axis)), 0.)).xyz;
    let radius = min(length(depth_axis), length(right_axis));
    //Change this for fully scaled arrows
    //out.radius = radius * 0.1;
    out.radius = radius * arrow_ampl * 0.1;

    let rotation_mat = mat3x3<f32>(
        right_axis,
        world_vector_arrow,
        depth_axis);

    var corrected_pos = model.position;
    //Change this for fully scaled arrows
    //corrected_pos.y = corrected_pos.y * arrow_ampl;
    corrected_pos = corrected_pos * arrow_ampl;

    let position = rotation_mat * corrected_pos * settings.magnitude + world_vector_pos;
    out.world_pos = position;
    let clip_pos = camera.view_proj * vec4<f32>(position, 1.0);
    out.clip_position = clip_pos + jitter.jitter * clip_pos.w;
    return out;
}

// function from :
// https://iquilezles.org/articles/intersectors/
fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }

fn iCappedCone(ro: vec3<f32>, rd: vec3<f32>,
                  pa: vec3<f32>, pb: vec3<f32>,
                  ra: f32, rb: f32 ) -> vec4<f32>
{
    let  ba = pb - pa;
    let  oa = ro - pa;
    let  ob = ro - pb;

    let m0 = dot(ba,ba);
    let m1 = dot(oa,ba);
    let m2 = dot(ob,ba);
    let m3 = dot(rd,ba);

    //caps
         if( m1<0.0 ) { if( dot2(oa*m3-rd*m1)<(ra*ra*m3*m3) ) { return vec4<f32>(-m1/m3,-ba*inverseSqrt(m0)); } }
    else if( m2>0.0 ) { if( dot2(ob*m3-rd*m2)<(rb*rb*m3*m3) ) { return vec4<f32>(-m2/m3, ba*inverseSqrt(m0)); } }

    // body
    let m4 = dot(rd,oa);
    let m5 = dot(oa,oa);
    let rr = ra - rb;
    let hy = m0 + rr*rr;

    let k2 = m0*m0    - m3*m3*hy;
    let k1 = m0*m0*m4 - m1*m3*hy + m0*ra*(rr*m3*1.0        );
    let k0 = m0*m0*m5 - m1*m1*hy + m0*ra*(rr*m1*2.0 - m0*ra);

    let h = k1*k1 - k2*k0;
    if( h<0.0 ) { return vec4(-1.0); }

    let t = (-k1-sqrt(h))/k2;

    let y = m1 + t*m3;
    if( y>0.0 && y<m0 )
    {
        return vec4<f32>(t, normalize(m0*(m0*(oa+t*rd)+rr*ba*ra)-ba*hy*y));
    }

    return vec4<f32>(-1.0);
}

fn cylIntersect( ro: vec3<f32>, rd: vec3<f32>, pa: vec3<f32>, pb: vec3<f32>, ra: f32 ) -> vec4<f32>
{
    let ba = pb-pa;
    let oc = ro - pa;
    let baba = dot(ba,ba);
    let bard = dot(ba,rd);
    let baoc = dot(ba,oc);
    let k2 = baba            - bard*bard;
    let k1 = baba*dot(oc,rd) - baoc*bard;
    let k0 = baba*dot(oc,oc) - baoc*baoc - ra*ra*baba;
    var h = k1*k1 - k2*k0;
    if( h<0.0 ) { return vec4<f32>(-1.0); }//no intersection
    h = sqrt(h);
    var t = (-k1-h)/k2;
    // body
    let y = baoc + t*bard;
    if( y>0.0 && y<baba ) { return vec4<f32>( t, (oc+t*rd - ba*y/baba)/ra ); }
    // caps
	t = ( select(baba, 0., y< 0.) - baoc) / bard;
    if( abs(k1+k2*t)<h )
    {
        return vec4<f32>( t, ba*sign(y)/sqrt(baba) );
    }
    return vec4<f32>(-1.0);//no intersection
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
}


@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let ro = camera.view_pos.xyz;
	let rd = normalize(in.world_pos - camera.view_pos.xyz);
    let pa = in.orig_position;
    let pb1 = in.orig_position + 0.5 * in.arrow;
    let pb2 = in.orig_position + in.arrow;

    var out: FragOutput;

    let traced_1 = iCappedCone(ro, rd, pb1, pb2, in.radius * settings.magnitude, 0.);
    let traced_2 = cylIntersect(ro, rd, pa, pb1, 0.5 * in.radius * settings.magnitude);
	if(max(traced_1.x, traced_2.x) < 0.) {
		discard;
	}
	let traced = select(traced_1, traced_2, traced_1.x < 0. || (traced_2.x < traced_1.x && traced_2.x > 0.));

	let pos = ro + traced.x * rd;
	let normal = traced.yzw;
	out.albedo = vec4<f32>(settings.color, 0.1);
    out.normal = vec4<f32>((normal + vec3<f32>(1.)) / 2. , 0.);
	let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}
";
