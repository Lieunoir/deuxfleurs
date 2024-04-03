use crate::texture;
use crate::util;
use wgpu::util::DeviceExt;

pub struct TextureCopy {
    square: wgpu::Buffer,
    copy_bind_group: wgpu::BindGroup,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    blend_bind_group: wgpu::BindGroup,
    copy_pipeline: wgpu::RenderPipeline,
    blend_pipeline: wgpu::RenderPipeline,
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

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
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

fn world_from_screen_coord(coord : vec2<f32>, depth_sample: f32) -> vec3<f32> {
    // reconstruct world-space position from the screen coordinate.
    let posClip = vec4(coord.x * 2.0 - 1.0, (1.0 - coord.y) * 2.0 - 1.0, depth_sample, 1.0);
    let posWorldW = camera.view_inv * posClip;
    let posWorld = posWorldW.xyz / posWorldW.www;
    return posWorld;
}

@fragment
fn fs_main(@builtin(position) fcoords : vec4<f32>) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(floor(fcoords.xy));
    let albedo   = textureLoad(t_a, coords, 0);
    if(albedo.w < 0.01) {
        discard;
    }
    //let position = textureLoad(t_pos, coords, 0).xyz;
    let buffer_size = textureDimensions(t_d);
    //let depth = textureLoad(t_d, coords, 0).x;
    let depth = textureSample(t_d, s, fcoords.xy / vec2<f32>(buffer_size)).x;
    let position = world_from_screen_coord(fcoords.xy / vec2<f32>(buffer_size), depth);
    let normal   = normalize(textureLoad(t_n, coords, 0).xyz * 2. - vec3<f32>(1.));
    //let normal   = textureLoad(t_n, coords, 0).xyz * 2. - vec3<f32>(1.);

	let view_dir = normalize(camera.view_pos.xyz - position);
    //let n_xy   = textureSample(t_n, in.tex_coords, 0).xy;
    //let n_z_t =  sqrt(1. - n_xy.x * n_xy.x - n_xy.y * n_xy.y);
    //let normal_t1 = vec3<f32>(n_xy, n_z_t);
    //let normal_t2 = vec3<f32>(n_xy, -n_z_t);
    //let n_z = select(n_z_t, -n_z_t, dot(normal_t2, view_dir) > dot(normal_t1, view_dir));
    ////let n_z = n_z_t;
    //let normal = vec3<f32>(n_xy, n_z);

	let light_dir = normalize(light.position - position);
	let half_dir = normalize(view_dir + light_dir);
	let F0 = vec3<f32>(0.04, 0.04, 0.04);
	let D = DistributionGGX(normal, half_dir, albedo.w);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, albedo.w);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	let kd = 1.0;
	let result = (kd * albedo.xyz + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);

    return vec4<f32>(result, 1.0);
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

    pub fn set_level(&mut self, queue: &mut wgpu::Queue, level: f32) {
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
        render_pass.set_bind_group(0, &camera_light_bind_group, &[]);
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
        render_pass.set_bind_group(0, &camera_light_bind_group, &[]);
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
        render_pass.set_bind_group(0, &camera_light_bind_group, &[]);
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
        render_pass.set_bind_group(0, &camera_light_bind_group, &[]);
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
    //let coords = in.clip_position.xy;
    //let weight = textureLoad(t_a, 0, coords);
    let weight = textureSample(t_a, s, in.tex_coords).x;
    //let buffer_size = vec2<f32>(textureDimensions(t_a));
    //var weight = 0.;
    //var tot_weight = 0.;
    //let radius = 5;
    ////let dpx = sqrt(dpdx(coords.x) * dpdx(coords.x) + dpdy(coords.x) * dpdy(coords.x)) * 200.;
    ////let dpy = sqrt(dpdx(coords.y) * dpdx(coords.y) + dpdy(coords.y) * dpdy(coords.y)) * 200.;
    //let dpx = buffer_size.x * camera.max_bb.x / 5.;
    //let dpy = buffer_size.y * camera.max_bb.y / 5.;
    //for (var r = 0; r < radius; r++) {
    //    for (var c = 0; c < radius; c++) {
    //        let w = exp((-f32(r*r)-f32(c*c))/(2.0*f32(radius*radius)));
    //        //let w = 1.;
    //        tot_weight += w;
    //        weight += w*textureSample(t_a, s, in.tex_coords + vec2<f32>(f32(r- radius/2) / dpx , f32(c- radius/2) / dpy)).x;
    //        //weight += textureSample(t_a, s, in.tex_coords + vec2<f32>(f32(r- radius/2) / dpx , f32(c- radius/2) / dpy)).x / f32(radius * radius);
    //    }
    //}
    //weight = weight / tot_weight;
    return vec4<f32>(0., 0., 0., 0.4 * weight);
}
";
