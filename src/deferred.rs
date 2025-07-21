use crate::texture;
use crate::util;
use crate::util::Vertex;
use wgpu::util::DeviceExt;

pub struct TextureCopy {
    square: wgpu::Buffer,
    copy_bind_group: wgpu::BindGroup,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    blend_bind_group: wgpu::BindGroup,
    copy_pipeline: wgpu::RenderPipeline,
    blend_pipeline: wgpu::RenderPipeline,
    screenshot_pipeline: wgpu::RenderPipeline,
    old_blend_texture: wgpu::Texture,
    old_blend_texture_view: wgpu::TextureView,
    new_blend_texture: wgpu::Texture,
    new_blend_texture_view: wgpu::TextureView,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SquareVertex {
    position: [f32; 3],
}

impl Vertex for SquareVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

impl TextureCopy {
    //resize
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("copy_texture"),
            view_formats: &[],
        };
        self.old_blend_texture = device.create_texture(&texture_descriptor);
        self.old_blend_texture_view = self
            .old_blend_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.new_blend_texture = device.create_texture(&texture_descriptor);
        self.new_blend_texture_view = self
            .new_blend_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.copy_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.old_blend_texture_view),
            }],
            label: Some("copy_bind_group"),
        });
        self.blend_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.copy_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.new_blend_texture_view),
            }],
            label: Some("copy_bind_group"),
        });
    }

    pub fn get_view(&self) -> &wgpu::TextureView {
        &self.new_blend_texture_view
    }

    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let positions = [[-1., -1., 0.], [1., -1., 0.], [-1., 1., 0.], [1., 1., 0.]];
        let vertices = positions.map(|position| SquareVertex { position });
        let square = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Copy Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture_descriptor = wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("copy_texture"),
            view_formats: &[],
        };
        let old_blend_texture = device.create_texture(&texture_descriptor);
        let old_blend_texture_view =
            old_blend_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let new_blend_texture = device.create_texture(&texture_descriptor);
        let new_blend_texture_view =
            new_blend_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let copy_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                }],
                label: Some("texture_bind_group_layout"),
            });

        let blend_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &copy_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&new_blend_texture_view),
            }],
            label: Some("blend_bind_group"),
        });

        let copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &copy_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&old_blend_texture_view),
            }],
            label: Some("copy_bind_group"),
        });
        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Copy Pipeline Layout"),
            bind_group_layouts: &[&copy_bind_group_layout],
            push_constant_ranges: &[],
        });
        let blend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blend Pipeline Layout"),
                bind_group_layouts: &[&copy_bind_group_layout],
                push_constant_ranges: &[],
            });
        let copy_shader = wgpu::ShaderModuleDescriptor {
            label: Some("copy shader"),
            source: wgpu::ShaderSource::Wgsl(COPY_SHADER.into()),
        };
        let copy_pipeline = util::create_copy_quad_pipeline(
            device,
            &copy_pipeline_layout,
            color_format,
            None,
            &[SquareVertex::desc()],
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            copy_shader.clone(),
            Some("copy render"),
        );
        let screenshot_pipeline = util::create_copy_quad_pipeline(
            device,
            &copy_pipeline_layout,
            crate::screenshot::SCREENSHOT_FORMAT,
            None,
            &[SquareVertex::desc()],
            Some(wgpu::BlendState::REPLACE),
            copy_shader.clone(),
            Some("copy render"),
        );
        let blend_pipeline = util::create_copy_quad_pipeline(
            device,
            &blend_pipeline_layout,
            color_format,
            None,
            &[SquareVertex::desc()],
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    dst_factor: wgpu::BlendFactor::Constant,
                    src_factor: wgpu::BlendFactor::OneMinusConstant,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    dst_factor: wgpu::BlendFactor::Constant,
                    src_factor: wgpu::BlendFactor::OneMinusConstant,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            copy_shader,
            Some("blend render"),
        );

        Self {
            square,
            copy_bind_group_layout,
            copy_bind_group,
            blend_bind_group,
            copy_pipeline,
            blend_pipeline,
            screenshot_pipeline,
            old_blend_texture,
            old_blend_texture_view,
            new_blend_texture,
            new_blend_texture_view,
        }
    }

    pub fn blend<'a, 'b>(&'a self, encoder: &mut wgpu::CommandEncoder, factor: f64, first: bool)
    where
        'a: 'b,
    {
        let load_op = if first {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            })
        } else {
            wgpu::LoadOp::Load
        };
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blend Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.old_blend_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_op,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_blend_constant(wgpu::Color {
            r: factor,
            g: factor,
            b: factor,
            a: factor,
        });
        render_pass.set_bind_group(0, &self.blend_bind_group, &[]);
        render_pass.set_pipeline(&self.blend_pipeline);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
    }

    pub fn copy<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(0, &self.copy_bind_group, &[]);
        render_pass.set_pipeline(&self.copy_pipeline);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
    }

    pub fn screenshot<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(0, &self.copy_bind_group, &[]);
        render_pass.set_pipeline(&self.screenshot_pipeline);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
    }
}

const COPY_SHADER: &str = "
@group(0) @binding(0)
var t_copy: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    ) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureLoad(t_copy, vec2<i32>(floor(in.clip_position.xy)), 0);
}
";

pub struct PBR {
    albedo: wgpu::Texture,
    albedo_view: wgpu::TextureView,
    normals: wgpu::Texture,
    normals_view: wgpu::TextureView,
    square: wgpu::Buffer,
    sampler: wgpu::Sampler,

    material_bind_group: wgpu::BindGroup,
    material_bind_group_layout: wgpu::BindGroupLayout,
    pbr_pipeline: wgpu::RenderPipeline,
}

impl PBR {
    pub fn get_albedo_view(&self) -> &wgpu::TextureView {
        &self.albedo_view
    }

    pub fn get_normals_view(&self) -> &wgpu::TextureView {
        &self.normals_view
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let descriptor = wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("pbr_texture"),
            view_formats: &[],
        };
        self.albedo = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("albedo_texture"),
            view_formats: &[],
        });
        self.albedo_view = self
            .albedo
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.normals = device.create_texture(&descriptor);
        self.normals_view = self
            .normals
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.normals_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
            label: Some("pbr_material_bind_group"),
        });
    }

    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_view: &wgpu::TextureView,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) -> Self {
        let positions = [[-1., -1., 0.], [1., -1., 0.], [-1., 1., 0.], [1., 1., 0.]];
        let vertices = positions.map(|position| SquareVertex { position });
        let square = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Copy Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let descriptor = wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("pbr_texture"),
            view_formats: &[],
        };
        let albedo = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("albedo_texture"),
            view_formats: &[],
        });
        let albedo_view = albedo.create_view(&wgpu::TextureViewDescriptor::default());
        let normals = device.create_texture(&descriptor);
        let normals_view = normals.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            //sample_type: wgpu::TextureSampleType::Depth,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
                label: Some("pbr_material_bind_group_layout"),
            });

        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normals_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("pbr_material_bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR Pipeline Layout"),
            bind_group_layouts: &[camera_light_bind_group_layout, &material_bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("pbr shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_SHADER.into()),
        };
        let pbr_pipeline = util::create_copy_quad_pipeline(
            device,
            &pipeline_layout,
            color_format,
            None,
            &[SquareVertex::desc()],
            None,
            shader,
            Some("pbr render"),
        );

        Self {
            albedo,
            albedo_view,
            normals,
            normals_view,
            square,
            sampler,

            material_bind_group,
            material_bind_group_layout,
            pbr_pipeline,
        }
    }

    pub fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(1, &self.material_bind_group, &[]);
        render_pass.set_pipeline(&self.pbr_pipeline);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
    }
}

const PBR_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

@group(1) @binding(0)
var t_a: texture_2d<f32>;
@group(1) @binding(1)
var t_n: texture_2d<f32>;
@group(1) @binding(2)
//var t_d: texture_depth_2d;
var t_d: texture_2d<f32>;
@group(1) @binding(3)
var s: sampler;


struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    ) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

const PI: f32 = 3.14159265359;

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

fn world_from_screen_coord(coord : vec2<f32>, depth_sample: f32) -> vec3<f32> {
    // reconstruct world-space position from the screen coordinate.
    let posClip = vec4(coord.x * 2.0 - 1.0, 1.0 - 2.0 * coord.y, depth_sample, 1.0);
    let posWorldW = camera.view_inv * posClip;
    let posWorld = posWorldW.xyz / posWorldW.www;
    return posWorld;
}

fn pcg3d(v_orig: vec3<u32>) -> vec3<u32> {
    var v = v_orig * 1664525 + 1013904223;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    v.x ^= v.x>>16u;
    v.y ^= v.y>>16u;
    v.z ^= v.z>>16u;

    v.x += v.y*v.z;
    v.y += v.z*v.x;
    v.z += v.x*v.y;

    return v;
}

fn fast_sqrt(x: f32) -> f32 {
    return f32(0x1FBD1DF5 + (u32(x) >> 1));
}

fn fast_acos(x: f32) -> f32 {
    var res = -0.156583 * abs(x) + PI / 2.0;
    res *= fast_sqrt(1. - abs(x));
    return select(PI - res, res, x >= 0);
}

const randoms = array<f32, 64>(0.9073287956637583, 0.8953753268762352, 0.3220086438462023, 0.007605212815564366, 0.01591998320496857, 0.16333876403470682, 0.7633080275109663, 0.6253689714158442, 0.9796289477520932, 0.47768855334816007, 0.20994347509627442, 0.42647190872472107, 0.3264460758651072, 0.603054743243745, 0.4421765326581557, 0.13635578498504275,
    0.5480187485794791, 0.7002945901365113, 0.04093307934142931, 0.8409299478779066, 0.3657819008493858, 0.3872717431211139, 0.5296179826887955, 0.3549791699992324, 0.03845149501235379, 0.9752711547848418, 0.20037853481683254, 0.31096408522103347, 0.9594224215818684, 0.9629871955616451, 0.4983265536276734, 0.002695323442428843,
    0.35680469302547124, 0.6338448300380964, 0.26924514548124223, 0.5489805045735846, 0.38712840331458065, 0.34813314754718905, 0.21110995223799223, 0.06735202851625521, 0.22925362499197766, 0.9693096630885775, 0.13104928603132715, 0.5136988570398621, 0.993335107309559, 0.8645336635925384, 0.05809545593417287, 0.12120304216110633,
    0.22041811198640138, 0.17310442191243958, 0.26970976141108405, 0.7577908143740093, 0.3530547214528106, 0.7158705393016846, 0.4373999583878948, 0.8503007357829833, 0.06923972709448556, 0.7685377089983041, 0.2800583414822193, 0.4926678074779679, 0.8794457785989035, 0.22453667177222958, 0.5565299827383392, 0.6752055012992703);

@fragment
fn fs_main(@builtin(position) fcoords : vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(floor(fcoords.xy));
    let albedo   = textureLoad(t_a, coords, 0);
    if(albedo.w < 0.01) {
        discard;
    }
    let buffer_size = textureDimensions(t_d);
    let depth = textureSample(t_d, s, fcoords.xy / vec2<f32>(buffer_size)).x;
    let position = world_from_screen_coord(fcoords.xy / vec2<f32>(buffer_size), depth);
    let normal   = normalize(textureLoad(t_n, coords, 0).xyz * 2. - vec3<f32>(1.));
	let view_dir = normalize(camera.view_pos.xyz - position);

    // SSAO
    //let rand = pcg3d(vec3<u32>(bitcast<u32>(coords.x), bitcast<u32>(coords.y), bitcast<u32>(depth)));
    //let randoms_i_1 = rand.x & 63;
    //let randoms_i_2 = rand.y & 63;
    //let randoms_i_3 = rand.z & 63;
    //let random_vec = vec3<f32>(
    //    2. * randoms[randoms_i_1] - 1.,
    //    2. * randoms[randoms_i_2] - 1.,
    //    2. * randoms[randoms_i_3] - 1.,
    //);
    //let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    //let bitangent = cross(normal, tangent);
    //let TBN = mat3x3<f32>(tangent, bitangent, normal);
    //var occlusion = 0.0;
    //let kernelSize: u32 = 6u;
    //let radius = 0.05;
    //for(var i: u32 = 0; i < kernelSize; i+=1)
    //{
    //    let rand2 = pcg3d(vec3<u32>(i, bitcast<u32>(depth), bitcast<u32>(coords.x * coords.y)));
    //    let angle_bias = 0.2;
    //    //let theta = (angle_bias + randoms[rand2.x & 63] * (1. - angle_bias)) * PI;
    //    let theta = (randoms[rand2.x & 63] - 0.5) * PI;
    //    let phi = randoms[rand2.y & 63] * PI;
    //    let i4 = rand2.z & 63;
    //    let offset = TBN * vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)); // from tangent to world
    //    let sample_pos = position + offset * radius * pow(randoms[i4], 2.);

    //    var sample = vec4(sample_pos, 1.0);
    //    sample = camera.view_proj * sample;    // from world to clip-space
    //    //let bias = 0.0025;
    //    let bias = 0.0000001;
    //    sample /= sample.w;               // perspective divide
    //    sample.x = sample.x * 0.5 + 0.5;
    //    sample.y = 0.5 - sample.y * 0.5;
    //    let sample_depth = sample.z;
    //    let origin = fcoords.xy / vec2<f32>(buffer_size);
    //    let sample_off = origin - sample.xy;
    //    let sampleDepth_1 = textureSample(t_d, s, origin + sample_off * 0.3).x;
    //    let sampleDepth_2 = textureSample(t_d, s, origin + sample_off * 0.6).x;
    //    let sampleDepth_3 = textureSample(t_d, s, sample.xy).x;
    //    let sample_world_pos_1 = world_from_screen_coord(origin + sample_off * 0.3, sampleDepth_1);
    //    let sample_world_pos_2 = world_from_screen_coord(origin + sample_off * 0.6, sampleDepth_2);
    //    let sample_world_pos_3 = world_from_screen_coord(sample.xy, sampleDepth_3);
    //    let range_check_1 = smoothstep(0.0, 1.0, radius / abs(sample_world_pos_1.z - position.z));
    //    let range_check_2 = smoothstep(0.0, 1.0, radius / abs(sample_world_pos_2.z - position.z));
    //    let range_check_3 = smoothstep(0.0, 1.0, radius / abs(sample_world_pos_3.z - position.z));
    //    let occlusion_1 = select(0., 1. / f32(kernelSize), sampleDepth_1 + bias < sample_depth) * range_check_1;
    //    let occlusion_2 = max(occlusion_1, select(0., 1. / f32(kernelSize), sampleDepth_2 + bias < sample_depth) * range_check_2);
    //    occlusion += max(occlusion_2, select(0., 1. / f32(kernelSize), sampleDepth_3 + bias < sample_depth) * range_check_3);
    //}

    // GTAO
    var visibility = 0.0;
    let kernelSize: u32 = 3u;
    let origin = fcoords.xy / vec2<f32>(buffer_size);
    let pix_dif = vec2<f32>(1.) / vec2<f32>(buffer_size);
    let camera_distance = sqrt(dot(camera.view_pos.xyz - position,camera.view_pos.xyz - position));
    let wanted_radius = 20. / camera_distance * pix_dif;
    let radius = vec2<f32>(
        min(wanted_radius.x, 20. * pix_dif.x),
        min(wanted_radius.y, 20. * pix_dif.y),
    );
    //let radius = min(min(0.005 / camera_distance, 30. * pix_dif.x), 30. * pix_dif.y);
    for(var i: u32 = 0; i < kernelSize; i+=1) {
        let rand = pcg3d(vec3<u32>(i, bitcast<u32>(depth), bitcast<u32>(coords.x * coords.y)));
        let phi = (randoms[rand.x & 63] + f32(i)) * PI / f32(kernelSize);
        let dir = vec2<f32>(cos(phi), -sin(phi)) * radius * randoms[rand.y & 63];
        //let dir = vec2<f32>(cos(phi), -sin(phi)) * radius;
        let world_dir = normalize(world_from_screen_coord(origin + dir, 0.) - position);
        let ortho_direction_v = world_dir - dot(world_dir, view_dir) * view_dir;
        let slice_plane_normal = normalize(cross(world_dir, view_dir));
        let projected_normal = normal - dot(normal, slice_plane_normal) * slice_plane_normal;
        let sign_n = sign(dot(ortho_direction_v, projected_normal));
        let cos_n = saturate(dot(view_dir,normalize(projected_normal)));
        let n = sign_n * acos(cos_n);


        let sample_coords_1 =  vec2<f32>( origin + 0.33 * dir);
        let sample_coords_2 =  vec2<f32>( origin + 0.66 * dir);
        let sample_coords_3 =  vec2<f32>( origin + 1.00 * dir);
        let sample_coords_1p =  vec2<f32>(origin - 0.33 * dir);
        let sample_coords_2p =  vec2<f32>(origin - 0.66 * dir);
        let sample_coords_3p =  vec2<f32>(origin - 1.00 * dir);
        let sample_depth_1 = textureSample(t_d, s,  sample_coords_1).x;
        let sample_depth_2 = textureSample(t_d, s,  sample_coords_2).x;
        let sample_depth_3 = textureSample(t_d, s,  sample_coords_3).x;
        let sample_depth_1p = textureSample(t_d, s, sample_coords_1p).x;
        let sample_depth_2p = textureSample(t_d, s, sample_coords_2p).x;
        let sample_depth_3p = textureSample(t_d, s, sample_coords_3p).x;

        let sample_1 =  world_from_screen_coord(sample_coords_1, sample_depth_1);
        let sample_2 =  world_from_screen_coord(sample_coords_2, sample_depth_2);
        let sample_3 =  world_from_screen_coord(sample_coords_3, sample_depth_3);
        let sample_1p = world_from_screen_coord(sample_coords_1p, sample_depth_1p);
        let sample_2p = world_from_screen_coord(sample_coords_2p, sample_depth_2p);
        let sample_3p = world_from_screen_coord(sample_coords_3p, sample_depth_3p);

        let d_s_1_squared = dot(sample_1  - position, sample_1 - position);
        let d_s_2_squared = dot(sample_2  - position, sample_2 - position);
        let d_s_3_squared = dot(sample_3  - position, sample_3 - position);
        let d_t_1_squared = dot(sample_1p - position, sample_1p - position);
        let d_t_2_squared = dot(sample_2p - position, sample_2p - position);
        let d_t_3_squared = dot(sample_3p - position, sample_3p - position);

        let d_s_1 = normalize(sample_1 - position);
        let d_s_2 = normalize(sample_2 - position);
        let d_s_3 = normalize(sample_3 - position);
        let d_t_1 = normalize(sample_1p - position);
        let d_t_2 = normalize(sample_2p - position);
        let d_t_3 = normalize(sample_3p - position);

        let world_radius = 10000.;
        //let world_radius = 0.5;
        let l_s_1 = saturate((sqrt(d_s_1_squared) - world_radius) / world_radius);
        let l_s_2 = saturate((sqrt(d_s_2_squared) - world_radius) / world_radius);
        let l_s_3 = saturate((sqrt(d_s_3_squared) - world_radius) / world_radius);
        let l_t_1 = saturate((sqrt(d_t_1_squared) - world_radius) / world_radius);
        let l_t_2 = saturate((sqrt(d_t_2_squared) - world_radius) / world_radius);
        let l_t_3 = saturate((sqrt(d_t_3_squared) - world_radius) / world_radius);
        let dot_s_1 = (1. - l_s_1) * dot(d_s_1, view_dir) + l_s_1 * cos(n + PI * 0.5);
        let dot_s_2 = (1. - l_s_2) * dot(d_s_2, view_dir) + l_s_2 * cos(n + PI * 0.5);
        let dot_s_3 = (1. - l_s_3) * dot(d_s_3, view_dir) + l_s_3 * cos(n + PI * 0.5);
        let dot_t_1 = (1. - l_t_1) * dot(d_t_1, view_dir) + l_t_1 * cos(n - PI * 0.5);
        let dot_t_2 = (1. - l_t_2) * dot(d_t_2, view_dir) + l_t_2 * cos(n - PI * 0.5);
        let dot_t_3 = (1. - l_t_3) * dot(d_t_3, view_dir) + l_t_3 * cos(n - PI * 0.5);

        let cos_h1 = max(max(dot_s_1, dot_s_2), dot_s_3);
        let cos_h2 = max(max(dot_t_1, dot_t_2), dot_t_3);
        let h1 = acos(cos_h1);
        let h2 = -acos(cos_h2);
        let h1p = n + clamp(h1 - n, -PI * 0.5, PI * 0.5);
        let h2p = n + clamp(h2 - n, -PI * 0.5, PI * 0.5);

        let projected_normal_length = sqrt(dot(projected_normal, projected_normal));
        let projected_normal_length_2 = 0.05 * projected_normal_length + (1. - 0.05);

        let local_visibility = 0.25 * projected_normal_length * (
            - cos(2. * h1p - n) + 2. * cos_n + 2. * (h1p + h2p) * sin(n)
            - cos(2. * h2p - n)
        );
        visibility += local_visibility / f32(kernelSize);
    }
    //visibility = 1.;

    let F0 = vec3<f32>(0.04, 0.04, 0.04);
    let kd = 1.;

	let up =  normalize(vec3<f32>(
	    camera.view_proj[0].y,
	    camera.view_proj[1].y,
		camera.view_proj[2].y
	));
	let right =  normalize(vec3<f32>(
	    camera.view_proj[0].x,
	    camera.view_proj[1].x,
		camera.view_proj[2].x
	));
	let forward =  normalize(vec3<f32>(
	    camera.view_proj[0].z,
	    camera.view_proj[1].z,
		camera.view_proj[2].z
	));
	let light_dir = normalize(right - up - forward);
	//let light_dir = normalize(light.position - position);
	let half_dir = normalize(view_dir + light_dir);
	let D = DistributionGGX(normal, half_dir, albedo.w);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, albedo.w);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	//var result = 0.55 * (kd * albedo.xyz * visibility + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
	var result = 0.5 * 0.55 * (kd * albedo.xyz + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);

	let light_dir_2 = normalize(-right + up - forward);
	//let light_dir_2 = normalize(vec3<f32>(1., 1., -1.));
	let half_dir_2 = normalize(view_dir + light_dir_2);
	let D2 = DistributionGGX(normal, half_dir_2, albedo.w);
	let F2 = fresnelSchlick(dot(half_dir_2, normal), F0);
	let G2 = GeometrySmith(normal, view_dir, light_dir_2, albedo.w);
	let f_ct_2 = D2 * F2 * G2 / (4. * dot(view_dir, normal) * dot(light_dir_2, normal));
	//result += 1.6 * (kd * albedo.xyz * visibility + PI * f_ct_2) * light.color * max(dot(normal, light_dir_2), 0.0);
	result += 0.5 * 1.6 * (kd * albedo.xyz + PI * f_ct_2) * light.color * max(dot(normal, light_dir_2), 0.0);

	let light_dir_3 = normalize(right + up + forward);
	let half_dir_3 = normalize(view_dir + light_dir_3);
	let D3 = DistributionGGX(normal, half_dir_3, albedo.w);
	let F3 = fresnelSchlick(dot(half_dir_3, normal), F0);
	let G3 = GeometrySmith(normal, view_dir, light_dir_3, albedo.w);
	let f_ct_3 = D3 * F3 * G3 / (4. * dot(view_dir, normal) * dot(light_dir_3, normal));
	//result += 1.4 * (kd * albedo.xyz * visibility + PI * f_ct_2) * light.color * max(dot(normal, light_dir_3), 0.0);
	result += 0.5 * 1.4 * (kd * albedo.xyz + PI * f_ct_2) * light.color * max(dot(normal, light_dir_3), 0.0);

	//result += 1.2 * kd * albedo.xyz * visibility ;
	//result *= 0.5;
	result *= 2. * visibility;

	//Tone mapping
	let m1 = mat3x3(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777,
    );
    let m2 = mat3x3(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602,
    );
    let v = m1 * result;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return vec4<f32>(clamp(m2 * (a / b), vec3(0.0), vec3(1.0)), 1.0);
    //return vec4<f32>(result, 1.0);
}
";

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GroundLevel {
    level: f32,
    padding: [f32; 3],
}

pub struct Ground {
    square: wgpu::Buffer,
    blur_pipeline: wgpu::RenderPipeline,
    h_blur_pipeline: wgpu::RenderPipeline,
    pipeline: wgpu::RenderPipeline,
    blurred_texture_view: wgpu::TextureView,
    h_blurred_texture_view: wgpu::TextureView,
    low_blurred_texture_view: wgpu::TextureView,
    low_h_blurred_texture_view: wgpu::TextureView,
    material_bind_group: wgpu::BindGroup,
    blur_bind_group: wgpu::BindGroup,
    h_blur_bind_group: wgpu::BindGroup,
    low_blur_bind_group: wgpu::BindGroup,
    low_h_blur_bind_group: wgpu::BindGroup,

    level_buffer: wgpu::Buffer,
    pub level: f32,
}

impl Ground {
    pub fn get_texture_view(&self) -> &wgpu::TextureView {
        &self.blurred_texture_view
    }

    pub fn set_level(&mut self, queue: &wgpu::Queue, level: f32) {
        self.level = level;
        let level = GroundLevel {
            level,
            padding: [0.; 3],
        };
        queue.write_buffer(&self.level_buffer, 0, bytemuck::cast_slice(&[level]));
    }

    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        _depth_view: &wgpu::TextureView,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        level: f32,
    ) -> Self {
        let level = GroundLevel {
            level,
            padding: [0.; 3],
        };
        let positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]];
        let vertices = positions.map(|position| SquareVertex { position });
        let square = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Shadow Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let size = wgpu::Extent3d {
            width: 1024,
            height: 1024,
            depth_or_array_layers: 1,
        };
        let low_size = wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Shadow texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture::Texture::SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let low_desc = wgpu::TextureDescriptor {
            label: Some("Shadow texture"),
            size: low_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture::Texture::SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let blurred_texture = device.create_texture(&desc);
        let h_blurred_texture = device.create_texture(&desc);
        let blurred_texture_view =
            blurred_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let h_blurred_texture_view =
            h_blurred_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let low_blurred_texture = device.create_texture(&low_desc);
        let low_h_blurred_texture = device.create_texture(&low_desc);
        let low_blurred_texture_view =
            low_blurred_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let low_h_blurred_texture_view =
            low_h_blurred_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("shadow_material_bind_group_layout"),
            });

        let material_ground_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("shadow_material_bind_group_layout"),
            });

        let h_blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blurred_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("blur_shadow_material_bind_group"),
        });

        let blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&h_blurred_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("blur_shadow_material_bind_group"),
        });
        let low_h_blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&low_blurred_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("blur_shadow_material_bind_group"),
        });

        let low_blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&low_h_blurred_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("blur_shadow_material_bind_group"),
        });

        let level_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ground Level Buffer"),
            contents: bytemuck::cast_slice(&[level]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &material_ground_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blurred_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: level_buffer.as_entire_binding(),
                },
            ],
            label: Some("shadow_material_bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[
                camera_light_bind_group_layout,
                &material_ground_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[camera_light_bind_group_layout, &material_bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("shadow shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        };

        let blur_shader = wgpu::ShaderModuleDescriptor {
            label: Some("shadow shader"),
            source: wgpu::ShaderSource::Wgsl(BLUR_SHADER.into()),
        };

        let h_blur_shader = wgpu::ShaderModuleDescriptor {
            label: Some("shadow shader"),
            source: wgpu::ShaderSource::Wgsl(H_BLUR_SHADER.into()),
        };
        let pipeline = util::create_double_sided_copy_quad_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[SquareVertex::desc()],
            Some(wgpu::BlendState::ALPHA_BLENDING),
            shader,
            Some("shadow render"),
        );

        let blur_pipeline = util::create_double_sided_copy_quad_pipeline(
            device,
            &blur_pipeline_layout,
            texture::Texture::SHADOW_FORMAT,
            None,
            &[SquareVertex::desc()],
            None,
            blur_shader,
            Some("blur shadow render"),
        );

        let h_blur_pipeline = util::create_double_sided_copy_quad_pipeline(
            device,
            &blur_pipeline_layout,
            texture::Texture::SHADOW_FORMAT,
            None,
            &[SquareVertex::desc()],
            None,
            h_blur_shader,
            Some("horizontal blur shadow render"),
        );

        Self {
            square,
            pipeline,
            blur_pipeline,
            h_blur_pipeline,
            blurred_texture_view,
            h_blurred_texture_view,
            low_blurred_texture_view,
            low_h_blurred_texture_view,
            material_bind_group,
            blur_bind_group,
            h_blur_bind_group,
            low_blur_bind_group,
            low_h_blur_bind_group,
            level_buffer,
            level: level.level,
        }
    }

    pub fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(1, &self.material_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
    }

    fn first_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_light_bind_group: &wgpu::BindGroup,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Horizontal Blur Shadow Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.low_h_blurred_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.h_blur_pipeline);
        render_pass.set_bind_group(0, camera_light_bind_group, &[]);
        render_pass.set_bind_group(1, &self.h_blur_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
        drop(render_pass);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blur Shadow Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.low_blurred_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.blur_pipeline);
        render_pass.set_bind_group(0, camera_light_bind_group, &[]);
        render_pass.set_bind_group(1, &self.low_blur_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
    }

    fn second_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_light_bind_group: &wgpu::BindGroup,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Horizontal Blur Shadow Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.h_blurred_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.h_blur_pipeline);
        render_pass.set_bind_group(0, camera_light_bind_group, &[]);
        render_pass.set_bind_group(1, &self.low_h_blur_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
        drop(render_pass);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blur Shadow Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.blurred_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_pipeline(&self.blur_pipeline);
        render_pass.set_bind_group(0, camera_light_bind_group, &[]);
        render_pass.set_bind_group(1, &self.blur_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.square.slice(..));
        render_pass.draw(0..4, 0..1);
    }

    pub fn blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera_light_bind_group: &wgpu::BindGroup,
    ) {
        self.first_pass(encoder, camera_light_bind_group);
        self.second_pass(encoder, camera_light_bind_group);
    }
}

const H_BLUR_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    min_bb: vec2<f32>,
    max_bb: vec2<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;
@group(1) @binding(0)
var t_a: texture_2d<f32>;
@group(1) @binding(1)
var s: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    ) -> VertexOutput {
    var out: VertexOutput;
    let clip_pos = vec4<f32>(model.position.x * 2. - 1., model.position.y * 2. - 1., 0., 1.);

    out.tex_coords = model.position.xy;
    out.clip_position = clip_pos;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    var OFFSET = array(0.0, 1.3846153846, 3.2307692308);
    var WEIGHT = array(0.2270270270, 0.3162162162, 0.0702702703);
    let buffer_size = vec2<f32>(textureDimensions(t_a));
    let coords = in.tex_coords;
    var weight = textureSample(t_a, s, coords).x * WEIGHT[0];
    let dpx = buffer_size.x;
    for (var r = 1; r < 3; r++) {
        weight += textureSample(t_a, s, coords + vec2<f32>(OFFSET[r] / dpx ,0.)).x * WEIGHT[r];
        weight += textureSample(t_a, s, coords - vec2<f32>(OFFSET[r] / dpx ,0.)).x * WEIGHT[r];
    }
    return weight;
}
";

const BLUR_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    min_bb: vec2<f32>,
    max_bb: vec2<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;
@group(1) @binding(0)
var t_a: texture_2d<f32>;
@group(1) @binding(1)
var s: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    ) -> VertexOutput {
    var out: VertexOutput;
    let clip_pos = vec4<f32>(model.position.x * 2. - 1., model.position.y * 2. - 1., 0., 1.);

    out.clip_position = clip_pos;
    out.tex_coords = model.position.xy;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    var OFFSET = array(0.0, 1.3846153846, 3.2307692308);
    var WEIGHT = array(0.2270270270, 0.3162162162, 0.0702702703);
    let buffer_size = vec2<f32>(textureDimensions(t_a));
    let coords = in.tex_coords;
    var weight = textureSample(t_a, s, coords).x * WEIGHT[0];
    let dpy = buffer_size.y;
    for (var r = 1; r < 3; r++) {
        weight += textureSample(t_a, s, coords + vec2<f32>(0., OFFSET[r] / dpy)).x * WEIGHT[r];
        weight += textureSample(t_a, s, coords - vec2<f32>(0., OFFSET[r] / dpy)).x * WEIGHT[r];
    }
    return weight;
}
";

const SHADOW_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    min_bb: vec2<f32>,
    max_bb: vec2<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

struct GroundLevel {
    level: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;
@group(1) @binding(0)
var t_a: texture_2d<f32>;
@group(1) @binding(1)
var s: sampler;
@group(1) @binding(2)
var<uniform> level: GroundLevel;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
    ) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = vec4<f32>(camera.min_bb.x + model.position.x * camera.max_bb.x, level.level, camera.min_bb.y + model.position.y * camera.max_bb.y, 1.);

    out.clip_position = camera.view_proj * world_pos;
    out.tex_coords = vec2<f32>(model.position.x, 1. - model.position.y);
    //out.clip_position = vec4<f32>(model.position, 1.);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let weight = textureSample(t_a, s, in.tex_coords).x;
    //These values are in linear color space and should NOT correspond to the screen output values
    return vec4<f32>(0., 0., 0., 0.4 * weight);
}
";
