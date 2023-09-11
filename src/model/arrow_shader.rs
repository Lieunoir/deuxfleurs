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
    _padding: vec3<u32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

@group(1) @binding(0)
var<uniform> transform: TransformUniform;
@group(2) @binding(0)
var<uniform> settings: SettingsUniform;
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) coords: vec3<f32>,
};

struct VectorInput {
    @location(2) color: vec3<f32>,
    @location(3) orig_position: vec3<f32>,
    @location(4) arrow: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
	@location(1) world_pos: vec3<f32>,
	@location(2) orig_position: vec3<f32>,
	@location(3) arrow: vec3<f32>,
	@location(4) radius: f32,
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
    let world_vector_arrow = (model_matrix * vec4<f32>(vector_i.orig_position + vector_i.arrow, 1.)).xyz - world_vector_pos;

    // We define the output we want to send over to frag shader
    var out: VertexOutput;

    out.color = vector_i.color;
	out.orig_position = world_vector_pos;
	out.arrow = world_vector_arrow * settings.magnitude;

    let view_axis = normalize(world_vector_pos - camera.view_pos.xyz);
    let arrow_axis = normalize(world_vector_arrow);
    let right_axis = (model_matrix * vec4<f32>(normalize(cross(view_axis, arrow_axis)), 0.)).xyz;
    let depth_axis = (model_matrix * vec4<f32>(-normalize(cross(arrow_axis, right_axis)), 0.)).xyz;
    let radius = min(length(depth_axis), length(right_axis));
    out.radius = radius;
    let right_axis = radius * normalize(cross(view_axis, arrow_axis));
    let depth_axis = -radius * normalize(cross(arrow_axis, right_axis));
    let rotation_mat = mat3x3<f32>(
        right_axis,
        world_vector_arrow,
        depth_axis);
    //let model_matrix_2 = mat3x3<f32>(model_matrix[0].xyz, model_matrix[1].xyz, model_matrix[2].xyz);
    //let position = rotation_mat * model_matrix_2 * model.position * settings.magnitude * 0.1 + world_vector_pos;
    let position = rotation_mat * model.position * settings.magnitude * 0.1 + world_vector_pos;
    out.world_pos = position;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
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

const PI: f32 = 3.14159;

// PBR functions taken from https://learnopengl.com/PBR/Theory
fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, a: f32) -> f32 {
    let a2     = a*a;
    let NdotH  = max(dot(N, H), 0.0);
    let NdotH2 = NdotH*NdotH;
	
    let nom    = a2;
    var denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;
	
    return nom / denom;
}

fn GeometrySchlickGGX(NdotV: f32, k: f32) -> f32
{
    let nom   = NdotV;
    let denom = NdotV * (1.0 - k) + k;
	
    return nom / denom;
}
  
fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, k: f32) -> f32
{
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = GeometrySchlickGGX(NdotV, k);
    let ggx2 = GeometrySchlickGGX(NdotL, k);
	
    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32>
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let ro = camera.view_pos.xyz;
	let rd = normalize(in.world_pos - camera.view_pos.xyz);
    let pa = in.orig_position;
    let pb1 = in.orig_position + 0.5 * in.arrow * 0.1;
    let pb2 = in.orig_position + in.arrow * 0.1;

    var out: FragOutput;

    let traced_1 = iCappedCone(ro, rd, pb1, pb2, 0.1 * 0.1 * in.radius * settings.magnitude, 0.);
    let traced_2 = cylIntersect(ro, rd, pa, pb1, 0.05 * 0.1 * in.radius * settings.magnitude);
	if(max(traced_1.x, traced_2.x) < 0.) {
		discard;
	}
	let traced = select(traced_1, traced_2, traced_1.x < 0. || (traced_2.x < traced_1.x && traced_2.x > 0.));
	
	let pos = ro + traced.x * rd;
	let normal = traced.yzw;
	let light_dir = normalize(light.position - pos);
	let view_dir = normalize(camera.view_pos.xyz - pos);
	let half_dir = normalize(view_dir + light_dir);
	let F0 = vec3<f32>(0.04, 0.04, 0.04);
	let D = DistributionGGX(normal, half_dir, 0.15);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, 0.01);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	let kd = 1.0;
	let lambertian = in.color;
	let result = (kd * lambertian + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
	out.color = vec4<f32>(result, 1.);
	let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	//out.depth = in.clip_position.z;
	return out;
}
";
