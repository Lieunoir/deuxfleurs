use wgpu::util::DeviceExt;
use crate::util;
use crate::texture;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BlendingUniform{
    factor: f32,
    _padding: [u32; 3],
}

pub struct TextureCopy {
    square: wgpu::Buffer,
    copy_bind_group: wgpu::BindGroup,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    blend_bind_group: wgpu::BindGroup,
    blend_bind_group_layout: wgpu::BindGroupLayout,
    copy_pipeline: wgpu::RenderPipeline,
    blend_pipeline: wgpu::RenderPipeline,
    copy_texture: wgpu::Texture,
    copy_texture_view: wgpu::TextureView,
    old_blend_texture: wgpu::Texture,
    old_blend_texture_view: wgpu::TextureView,
    new_blend_texture: wgpu::Texture,
    new_blend_texture_view: wgpu::TextureView,
    copy_sampler: wgpu::Sampler,
    blend_factor_buffer: wgpu::Buffer,
    texture_size: wgpu::Extent3d,
    pub dirty: bool,
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
        height: u32) {
        self.texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        self.copy_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: self.texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                label: Some("copy_texture"),
                view_formats: &[],
            }
        );
        self.copy_texture_view = self.copy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.old_blend_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: self.texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("copy_old_texture"),
                view_formats: &[],
            }
        );
        self.old_blend_texture_view = self.old_blend_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.new_blend_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: self.texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("copy_old_texture"),
                view_formats: &[],
            }
        );
        self.new_blend_texture_view = self.new_blend_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.copy_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &self.copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.copy_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.copy_sampler),
                    },
                ],
                label: Some("copy_bind_group"),
            }
        );
        self.blend_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &self.blend_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.new_blend_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.old_blend_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.copy_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.blend_factor_buffer.as_entire_binding(),
                    },
                ],
                label: Some("copy_bind_group"),
            }
        );
    }

    pub fn get_view(&self) -> &wgpu::TextureView {
        &self.new_blend_texture_view
    }

    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32
    ) -> Self {
        let positions = [
            [-1., -1., 0.],
            [1., -1., 0.],
            [-1., 1., 0.],
            [1., 1., 0.],
        ];
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
        let copy_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                label: Some("copy_texture"),
                view_formats: &[],
            }
        );
        let copy_texture_view = copy_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let old_blend_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("copy_old_texture"),
                view_formats: &[],
            }
        );
        let old_blend_texture_view = old_blend_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let new_blend_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1, // We'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("copy_old_texture"),
                view_formats: &[],
            }
        );
        let new_blend_texture_view = new_blend_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let copy_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let blend_bind_group_layout =
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
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let copy_bind_group_layout =
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
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let blending = BlendingUniform {
            factor: 0.,
            _padding: [0; 3],
        };
        let blend_factor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Blend Factor Buffer"),
            contents: bytemuck::cast_slice(&[blending]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let blend_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &blend_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&new_blend_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&old_blend_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&copy_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: blend_factor_buffer.as_entire_binding(),
                    },
                ],
                label: Some("blend_bind_group"),
            }
        );

        let copy_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&copy_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&copy_sampler),
                    },
                ],
                label: Some("copy_bind_group"),
            }
        );

        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Copy Pipeline Layout"),
            bind_group_layouts: &[&copy_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blend_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Copy Pipeline Layout"),
            bind_group_layouts: &[&blend_bind_group_layout],
            push_constant_ranges: &[],
        });

        let copy_shader = wgpu::ShaderModuleDescriptor {
            label: Some("copy shader"),
            source: wgpu::ShaderSource::Wgsl(COPY_SHADER.into()),
        };
        let blend_shader = wgpu::ShaderModuleDescriptor {
            label: Some("blend shader"),
            source: wgpu::ShaderSource::Wgsl(BLEND_SHADER.into()),
        };
        let copy_pipeline = util::create_copy_quad_pipeline(
            device,
            &copy_pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[SquareVertex::desc()],
            copy_shader,
            Some("copy render"),
        );
        let blend_pipeline = util::create_copy_quad_pipeline(
            device,
            &blend_pipeline_layout,
            color_format,
            None,
            &[SquareVertex::desc()],
            blend_shader,
            Some("blend render"),
        );


        Self {
            square,
            copy_bind_group,
            copy_bind_group_layout,
            blend_bind_group,
            blend_bind_group_layout,
            copy_pipeline,
            blend_pipeline,
            copy_texture,
            copy_texture_view,
            old_blend_texture,
            old_blend_texture_view,
            new_blend_texture,
            new_blend_texture_view,
            copy_sampler,
            blend_factor_buffer,
            texture_size,
            dirty: true,
        }
    }

    pub fn blend<'a, 'b>(&'a self, queue: &mut wgpu::Queue, encoder: &mut wgpu::CommandEncoder, factor: f32)
    where
        'a: 'b,
    {
        let blending = BlendingUniform {
            factor,
            _padding: [0; 3],
        };
        queue.write_buffer(
            &self.blend_factor_buffer,
            0,
            bytemuck::cast_slice(&[blending]),
        );
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blend Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.copy_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            // Create a depth stencil buffer using the depth texture
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
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

    pub fn save_old(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.copy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.old_blend_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            self.texture_size,
        );
    }
}

const COPY_SHADER: &str = "
@group(0) @binding(0)
var t_copy: texture_2d<f32>;
@group(0) @binding(1)
var s_copy: sampler;


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
    out.tex_coords = vec2<f32>(model.position.x * 0.5 + 0.5, 1. - model.position.y * 0.5 - 0.5);
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_copy, s_copy, in.tex_coords);
}
";

const BLEND_SHADER: &str = "

struct BlendFactor {
    factor: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0)
var t_copy: texture_2d<f32>;
@group(0) @binding(1)
var t_copy_old: texture_2d<f32>;
@group(0) @binding(2)
var s_copy: sampler;
@group(0) @binding(3)
var<uniform> blend_factor: BlendFactor;


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
    out.tex_coords = vec2<f32>(model.position.x * 0.5 + 0.5, 1. - model.position.y * 0.5 - 0.5);
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let new_frame = textureSample(t_copy, s_copy, in.tex_coords);
    let old_frame = textureSample(t_copy_old, s_copy, in.tex_coords);
    return blend_factor.factor * old_frame + (1. - blend_factor.factor) * new_frame;
}
";

pub struct PBR {
    positions: wgpu::Texture,
    positions_view: wgpu::TextureView,
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
    pub fn get_positions_view(&self) -> &wgpu::TextureView {
        &self.positions_view
    }

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
        width: u32,
        height: u32) {
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
        self.positions = device.create_texture( &
            wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("positions_texture"),
                view_formats: &[],
            });
        self.positions_view = self.positions.create_view(&wgpu::TextureViewDescriptor::default());
        self.albedo = device.create_texture( &
            wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: color_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("albedo_texture"),
                view_formats: &[],
            });
        self.albedo_view = self.albedo.create_view(&wgpu::TextureViewDescriptor::default());
        self.normals = device.create_texture(&descriptor);
        self.normals_view = self.normals.create_view(&wgpu::TextureViewDescriptor::default());

        self.material_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &self.material_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.positions_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.normals_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    }
                ],
                label: Some("pbr_material_bind_group"),
            }
        );
    }

    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        width: u32,
        height: u32
    ) -> Self {
        let positions = [
            [-1., -1., 0.],
            [1., -1., 0.],
            [-1., 1., 0.],
            [1., 1., 0.],
        ];
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
        let positions = device.create_texture( &
            wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("positions_texture"),
                view_formats: &[],
            });
        let positions_view = positions.create_view(&wgpu::TextureViewDescriptor::default());
        let albedo = device.create_texture( &
            wgpu::TextureDescriptor {
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

        let material_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &material_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&positions_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&albedo_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&normals_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    }
                ],
                label: Some("pbr_material_bind_group"),
            }
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR Pipeline Layout"),
            bind_group_layouts: &[
                camera_light_bind_group_layout,
                &material_bind_group_layout,
            ],
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
            Some(texture::Texture::DEPTH_FORMAT),
            &[SquareVertex::desc()],
            shader,
            Some("pbr render"),
        );

        Self {
            positions,
            positions_view,
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
var t_pos: texture_2d<f32>;
@group(1) @binding(1)
var t_a: texture_2d<f32>;
@group(1) @binding(2)
var t_n: texture_2d<f32>;
@group(1) @binding(3)
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
    out.tex_coords = vec2<f32>(model.position.x * 0.5 + 0.5, 1. - model.position.y * 0.5 - 0.5);
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let albedo   = textureSample(t_a  , s, in.tex_coords);
    if(albedo.w < 0.01) {
        discard;
    }
    let position = textureSample(t_pos, s, in.tex_coords).xyz;
    let normal   = normalize(textureSample(t_n  , s, in.tex_coords).xyz * 2. - vec3<f32>(1.));
    //let normal   = textureSample(t_n  , s, in.tex_coords).xyz * 2. - vec3<f32>(1.);

	let view_dir = normalize(camera.view_pos.xyz - position);
    //let n_xy   = textureSample(t_n  , s, in.tex_coords).xy;
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
