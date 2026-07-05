use crate::camera::Camera;
use crate::texture;
use crate::util;
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;
#[cfg(feature = "profiling")]
use wgpu_profiler::GpuProfiler;

pub struct TextureCopy {
    copy_bind_group: wgpu::BindGroup,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    blend_bind_group: wgpu::BindGroup,
    copy_pipeline: wgpu::RenderPipeline,
    blend_pipeline: wgpu::RenderPipeline,
}

impl TextureCopy {
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        blend_stored_view: &wgpu::TextureView,
        blend_render_target_view: &wgpu::TextureView,
    ) {
        self.copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.copy_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(blend_stored_view),
            }],
            label: Some("copy_bind_group"),
        });
        self.blend_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.copy_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(blend_render_target_view),
            }],
            label: Some("blend_bind_group"),
        });
    }

    pub fn new(
        device: &wgpu::Device,
        blend_stored_view: &wgpu::TextureView,
        blend_render_target_view: &wgpu::TextureView,
        color_format: wgpu::TextureFormat,
    ) -> Self {
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
                resource: wgpu::BindingResource::TextureView(&blend_render_target_view),
            }],
            label: Some("blend_bind_group"),
        });

        let copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &copy_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&blend_stored_view),
            }],
            label: Some("copy_bind_group"),
        });
        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Copy Pipeline Layout"),
            bind_group_layouts: &[Some(&copy_bind_group_layout)],
            immediate_size: 0,
        });
        let blend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blend Pipeline Layout"),
                bind_group_layouts: &[Some(&copy_bind_group_layout)],
                immediate_size: 0,
            });
        let copy_shader = include_wgsl!("copy.wgsl");
        let copy_pipeline = util::create_copy_quad_pipeline(
            device,
            &copy_pipeline_layout,
            color_format,
            None,
            &[],
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            copy_shader.clone(),
            Some("copy render"),
        );
        let blend_pipeline = util::create_copy_quad_pipeline(
            device,
            &blend_pipeline_layout,
            texture::SCREENSHOT_FORMAT,
            None,
            &[],
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
            copy_bind_group_layout,
            copy_bind_group,
            blend_bind_group,
            copy_pipeline,
            blend_pipeline,
        }
    }

    pub fn blend<'a, 'b>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        factor: f64,
        first: bool,
        blend_stored_view: &wgpu::TextureView,
    ) where
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
                view: blend_stored_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_op,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_blend_constant(wgpu::Color {
            r: factor,
            g: factor,
            b: factor,
            a: factor,
        });
        render_pass.set_bind_group(0, &self.blend_bind_group, &[]);
        render_pass.set_pipeline(&self.blend_pipeline);
        render_pass.draw(0..4, 0..1);
    }

    pub fn copy<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(0, &self.copy_bind_group, &[]);
        render_pass.set_pipeline(&self.copy_pipeline);
        render_pass.draw(0..4, 0..1);
    }
}

pub struct PBR {
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    material_bind_group_ping: wgpu::BindGroup,
    material_bind_group_pong: wgpu::BindGroup,
    material_bind_group_layout: wgpu::BindGroupLayout,
    pbr_pipeline: wgpu::RenderPipeline,
}

impl PBR {
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        camera: &Camera,
        queue: &wgpu::Queue,
        albedo_view: &wgpu::TextureView,
        normals_view: &wgpu::TextureView,
        denoised_ssao_view_ping: &wgpu::TextureView,
        denoised_ssao_view_pong: &wgpu::TextureView,
    ) {
        let (x_mul, x_add) = camera.get_uv_to_view_x_mul_add();
        let (y_mul, y_add) = camera.get_uv_to_view_y_mul_add();
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::cast_slice(&[x_mul, x_add, y_mul, y_add]),
        );
        self.material_bind_group_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normals_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(denoised_ssao_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params.as_entire_binding(),
                },
            ],
            label: Some("pbr_material_bind_group_ping"),
        });
        self.material_bind_group_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normals_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(denoised_ssao_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params.as_entire_binding(),
                },
            ],
            label: Some("pbr_material_bind_group_pong"),
        });
    }

    pub fn new(
        device: &wgpu::Device,
        camera: &Camera,
        color_format: wgpu::TextureFormat,
        albedo_view: &wgpu::TextureView,
        normals_view: &wgpu::TextureView,
        denoised_ssao_view_ping: &wgpu::TextureView,
        denoised_ssao_view_pong: &wgpu::TextureView,
    ) -> Self {
        let (x_mul, x_add) = camera.get_uv_to_view_x_mul_add();
        let (y_mul, y_add) = camera.get_uv_to_view_y_mul_add();

        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PBR Params Buffer"),
            contents: bytemuck::cast_slice(&[x_mul, x_add, y_mul, y_add]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("pbr_material_bind_group_layout"),
            });

        let material_bind_group_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: wgpu::BindingResource::TextureView(denoised_ssao_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
            label: Some("pbr_material_bind_group_ping"),
        });
        let material_bind_group_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: wgpu::BindingResource::TextureView(denoised_ssao_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
            label: Some("pbr_material_bind_group_pong"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PBR Pipeline Layout"),
            bind_group_layouts: &[Some(&material_bind_group_layout)],
            immediate_size: 0,
        });

        let shader = include_wgsl!("pbr.wgsl");
        let pbr_pipeline = util::create_copy_quad_pipeline(
            device,
            &pipeline_layout,
            color_format,
            None,
            &[],
            None,
            shader,
            Some("pbr render"),
        );

        Self {
            params,
            sampler,
            material_bind_group_ping,
            material_bind_group_pong,
            material_bind_group_layout,
            pbr_pipeline,
        }
    }

    pub fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>, ping: bool)
    where
        'a: 'b,
    {
        if ping {
            render_pass.set_bind_group(0, &self.material_bind_group_ping, &[]);
        } else {
            render_pass.set_bind_group(0, &self.material_bind_group_pong, &[]);
        }
        render_pass.set_pipeline(&self.pbr_pipeline);
        render_pass.draw(0..4, 0..1);
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GroundLevel {
    level: f32,
    padding: [f32; 3],
}

pub struct Ground {
    blur_pipeline: wgpu::RenderPipeline,
    h_blur_pipeline: wgpu::RenderPipeline,
    pipeline: wgpu::RenderPipeline,
    // Using multiple texture but at lower res + only u8,
    // so in total less than a render target
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
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        level: f32,
    ) -> Self {
        let level = GroundLevel {
            level,
            padding: [0.; 3],
        };

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
            format: texture::SHADOW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let low_desc = wgpu::TextureDescriptor {
            label: Some("Shadow texture"),
            size: low_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture::SHADOW_FORMAT,
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
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
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
                Some(camera_bind_group_layout),
                Some(&material_ground_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[Some(&material_bind_group_layout)],
            immediate_size: 0,
        });

        let shader = include_wgsl!("shadow.wgsl");
        let blur_shader = include_wgsl!("blur.wgsl");
        let shader = device.create_shader_module(shader);
        let blur_shader = device.create_shader_module(blur_shader);
        let pipeline = util::create_double_sided_copy_quad_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::DEPTH_FORMAT),
            &[],
            Some(wgpu::BlendState::ALPHA_BLENDING),
            "fs_main",
            &shader,
            Some("shadow render"),
        );

        let blur_pipeline = util::create_double_sided_copy_quad_pipeline(
            device,
            &blur_pipeline_layout,
            texture::SHADOW_FORMAT,
            None,
            &[],
            None,
            "vertical_fs_main",
            &blur_shader,
            Some("blur shadow render"),
        );

        let h_blur_pipeline = util::create_double_sided_copy_quad_pipeline(
            device,
            &blur_pipeline_layout,
            texture::SHADOW_FORMAT,
            None,
            &[],
            None,
            "horizontal_fs_main",
            &blur_shader,
            Some("horizontal blur shadow render"),
        );

        Self {
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
        render_pass.draw(0..4, 0..1);
    }

    fn first_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Horizontal Blur Shadow Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.low_h_blurred_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.h_blur_pipeline);
        render_pass.set_bind_group(0, &self.h_blur_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
        drop(render_pass);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blur Shadow Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.low_blurred_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.blur_pipeline);
        render_pass.set_bind_group(0, &self.low_blur_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }

    fn second_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Horizontal Blur Shadow Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.h_blurred_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.h_blur_pipeline);
        render_pass.set_bind_group(0, &self.low_h_blur_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
        drop(render_pass);
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blur Shadow Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.blurred_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&self.blur_pipeline);
        render_pass.set_bind_group(0, &self.blur_bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }

    pub fn blur(&self, encoder: &mut wgpu::CommandEncoder) {
        self.first_pass(encoder);
        self.second_pass(encoder);
    }
}

fn create_hilbert_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    const LEVEL: usize = 4;
    //const TEXTURE_ROW_SIZE: usize = 2usize.pow(LEVEL as u32);
    //const TEXTURE_COL_SIZE: usize = 2usize.pow(LEVEL as u32);
    const TEXTURE_ROW_SIZE: usize = 16;
    const TEXTURE_COL_SIZE: usize = 16;
    let texture_size = wgpu::Extent3d {
        width: TEXTURE_ROW_SIZE as u32,
        height: TEXTURE_COL_SIZE as u32,
        depth_or_array_layers: 1,
    };
    let hilbert_noise = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: Some("hilbert_noise"),
        view_formats: &[],
    });
    const BYTE_PER_ENTRY: usize = size_of::<u8>();

    let mut buffer = [0; TEXTURE_ROW_SIZE * TEXTURE_COL_SIZE * BYTE_PER_ENTRY];
    for i in 0..TEXTURE_COL_SIZE {
        for j in 0..TEXTURE_ROW_SIZE {
            let mut p = (i, j);
            let mut d = 0;
            for k in 0..LEVEL {
                let n_i = LEVEL - k - 1;
                let n = n_i as u32;
                let r = ((p.0 >> n) & 1, (p.1 >> n) & 1);
                d += ((3 * r.0) ^ r.1) << (2 * n);
                if r.1 == 0 {
                    if r.0 == 1 {
                        p.0 = (1usize << n).wrapping_sub(1).wrapping_sub(p.0);
                        p.1 = (1usize << n).wrapping_sub(1).wrapping_sub(p.1);
                    }
                    let temp = p.0;
                    p.0 = p.1;
                    p.1 = temp;
                }
            }
            for k in 0..BYTE_PER_ENTRY {
                buffer[(i * TEXTURE_ROW_SIZE + j) as usize * BYTE_PER_ENTRY + k] =
                    d.to_ne_bytes()[k];
            }
        }
    }

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &hilbert_noise,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &buffer,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(BYTE_PER_ENTRY as u32 * TEXTURE_ROW_SIZE as u32),
            rows_per_image: Some(TEXTURE_COL_SIZE as u32),
        },
        texture_size,
    );
    hilbert_noise.create_view(&wgpu::TextureViewDescriptor::default())
}

pub struct SSAO {
    frame_index: u32,
    ssao_commons: wgpu::Buffer,
    hilbert_noise: wgpu::TextureView,
    sampler: wgpu::Sampler,
    noise_sampler: wgpu::Sampler,
    depth_mip_sampler: wgpu::Sampler,
    depth_bind_group_ping: wgpu::BindGroup,
    depth_bind_group_pong: wgpu::BindGroup,
    depth_bind_group_layout: wgpu::BindGroupLayout,
    ssao_pipeline: wgpu::RenderPipeline,
    denoiser_bind_group_ping: wgpu::BindGroup,
    denoiser_bind_group_pong: wgpu::BindGroup,
    denoiser_bind_group_layout: wgpu::BindGroupLayout,
    denoiser_pipeline: wgpu::RenderPipeline,
    depth_copy_bind_group_layout: wgpu::BindGroupLayout,
    depth_copy_bind_group: wgpu::BindGroup,
    depth_copy_pipeline: wgpu::RenderPipeline,
    depth_filter_bind_group_layout: wgpu::BindGroupLayout,
    depth_filter_bind_groups_ping:
        [wgpu::BindGroup; texture::FILTERED_DEPTH_MIP_LEVEL_COUNT as usize - 1],
    depth_filter_bind_groups_pong:
        [wgpu::BindGroup; texture::FILTERED_DEPTH_MIP_LEVEL_COUNT as usize - 1],
    depth_filter_pipeline: wgpu::RenderPipeline,
    cleared: bool,
    ping: bool,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SSAOParams {
    reprojection: [[f32; 4]; 4],
    depth_mul: f32,
    depth_add: f32,
    x_mul: f32,
    x_add: f32,
    y_mul: f32,
    y_add: f32,
    frame_index: u32,
    world_distance_inv: f32,
    pix_size: [f32; 2],
    _pad0: u32,
    _pad1: u32,
}

impl SSAO {
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        normals_view: &wgpu::TextureView,
        ssao_view: &wgpu::TextureView,
        denoiser_edges_view: &wgpu::TextureView,
        denoised_ssao_view_ping: &wgpu::TextureView,
        denoised_ssao_view_pong: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        filtered_depth_view_ping: &wgpu::TextureView,
        filtered_depth_mip_views_ping: &[wgpu::TextureView;
             texture::FILTERED_DEPTH_MIP_LEVEL_COUNT as usize],
        filtered_depth_view_pong: &wgpu::TextureView,
        filtered_depth_mip_views_pong: &[wgpu::TextureView;
             texture::FILTERED_DEPTH_MIP_LEVEL_COUNT as usize],
    ) {
        self.cleared = false;
        self.depth_bind_group_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.depth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(normals_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.depth_mip_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ssao_commons.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.hilbert_noise),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
            ],
            label: Some("ssao_bind_group"),
        });
        self.depth_bind_group_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.depth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(normals_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.depth_mip_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.ssao_commons.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.hilbert_noise),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.noise_sampler),
                },
            ],
            label: Some("ssao_bind_group"),
        });
        self.denoiser_bind_group_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.denoiser_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(denoiser_edges_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(denoised_ssao_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.ssao_commons.as_entire_binding(),
                },
            ],
            label: Some("denoiser_bind_group_ping"),
        });
        self.denoiser_bind_group_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.denoiser_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(denoiser_edges_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(denoised_ssao_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.ssao_commons.as_entire_binding(),
                },
            ],
            label: Some("denoiser_bind_group_pong"),
        });
        self.depth_copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.depth_copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.ssao_commons.as_entire_binding(),
                },
            ],
            label: Some("depth_copy_bind_group"),
        });
        self.depth_filter_bind_groups_ping = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.depth_filter_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &filtered_depth_mip_views_ping[i],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
                label: Some("depth_filter_bind_group"),
            })
        });
        self.depth_filter_bind_groups_pong = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.depth_filter_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &filtered_depth_mip_views_pong[i],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
                label: Some("depth_filter_bind_group"),
            })
        });
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        normals_view: &wgpu::TextureView,
        ssao_view: &wgpu::TextureView,
        denoiser_edges_view: &wgpu::TextureView,
        denoised_ssao_view_ping: &wgpu::TextureView,
        denoised_ssao_view_pong: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        filtered_depth_view_ping: &wgpu::TextureView,
        filtered_depth_mip_views_ping: &[wgpu::TextureView;
             texture::FILTERED_DEPTH_MIP_LEVEL_COUNT as usize],
        filtered_depth_view_pong: &wgpu::TextureView,
        filtered_depth_mip_views_pong: &[wgpu::TextureView;
             texture::FILTERED_DEPTH_MIP_LEVEL_COUNT as usize],
    ) -> Self {
        let frame_index = 0;
        let params = SSAOParams {
            reprojection: [[0.; 4]; 4],
            depth_mul: 1.,
            depth_add: 0.,
            x_mul: 1.,
            x_add: 0.,
            y_mul: 1.,
            y_add: 0.,
            frame_index: 289 * (frame_index % 64),
            world_distance_inv: 1.,
            pix_size: [1.; 2],
            _pad0: 0,
            _pad1: 0,
        };
        let ssao_commons = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSAO Common Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let depth_mip_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let depth_bind_group_layout =
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Uint,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
                label: Some("ssao_bind_group_layout"),
            });

        let denoiser_bind_group_layout =
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
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("denoiser_bind_group_layout"),
            });

        let hilbert_noise = create_hilbert_texture(device, queue);
        let depth_bind_group_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &depth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(normals_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&depth_mip_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ssao_commons.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&hilbert_noise),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&noise_sampler),
                },
            ],
            label: Some("ssao_bind_group_ping"),
        });

        let depth_bind_group_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &depth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(normals_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&depth_mip_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ssao_commons.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&hilbert_noise),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&noise_sampler),
                },
            ],
            label: Some("ssao_bind_group_pong"),
        });

        let denoiser_bind_group_ping = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &denoiser_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(denoiser_edges_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(denoised_ssao_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: ssao_commons.as_entire_binding(),
                },
            ],
            label: Some("denoiser_bind_group_ping"),
        });

        let denoiser_bind_group_pong = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &denoiser_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(denoiser_edges_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(denoised_ssao_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_pong),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(filtered_depth_view_ping),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: ssao_commons.as_entire_binding(),
                },
            ],
            label: Some("denoiser_bind_group_pong"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAO Pipeline Layout"),
            bind_group_layouts: &[Some(&depth_bind_group_layout)],
            immediate_size: 0,
        });

        let (shader, denoiser_shader) = if device.features().contains(wgpu::Features::SHADER_F16) {
            (
                device.create_shader_module(include_wgsl!("ssao_f16.wgsl")),
                include_wgsl!("denoiser_f16.wgsl"),
            )
        } else {
            (
                device.create_shader_module(include_wgsl!("ssao.wgsl")),
                include_wgsl!("denoiser.wgsl"),
            )
        };

        let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao_pipeline"),
            layout: Some(&pipeline_layout),
            multiview_mask: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: texture::SSAO_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: texture::SSAO_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[(
                        "max_mip_level",
                        (texture::FILTERED_DEPTH_MIP_LEVEL_COUNT - 1) as f64,
                    )],
                    ..Default::default()
                },
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
        });

        let denoiser_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSAO denoiser Pipeline Layout"),
                bind_group_layouts: &[Some(&denoiser_bind_group_layout)],
                immediate_size: 0,
            });

        let denoiser_pipeline = util::create_copy_quad_pipeline(
            device,
            &denoiser_pipeline_layout,
            texture::SSAO_FORMAT,
            None,
            &[],
            None,
            denoiser_shader,
            Some("ssao denoiser render"),
        );

        let depth_copy_bind_group_layout =
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
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("depth_copy_bind_group_layout"),
            });
        let depth_copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &depth_copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ssao_commons.as_entire_binding(),
                },
            ],
            label: Some("depth_copy_bind_group"),
        });
        let depth_copy_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Depth copy Pipeline Layout"),
                bind_group_layouts: &[Some(&depth_copy_bind_group_layout)],
                immediate_size: 0,
            });
        let depth_copy_shader = include_wgsl!("copy_depth.wgsl");
        let depth_copy_pipeline = util::create_copy_quad_pipeline(
            device,
            &depth_copy_pipeline_layout,
            texture::FILTERED_DEPTH_FORMAT,
            None,
            &[],
            None,
            depth_copy_shader,
            Some("depth copy render"),
        );

        let depth_filter_bind_group_layout =
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("depth_filter_bind_group_layout"),
            });
        let depth_filter_bind_groups_ping = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &depth_filter_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &filtered_depth_mip_views_ping[i],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: Some("depth_filter_bind_group_ping"),
            })
        });
        let depth_filter_bind_groups_pong = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &depth_filter_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &filtered_depth_mip_views_pong[i],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: Some("depth_filter_bind_group_pong"),
            })
        });
        let depth_filter_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Depth filter Pipeline Layout"),
                bind_group_layouts: &[Some(&depth_filter_bind_group_layout)],
                immediate_size: 0,
            });
        let depth_filter_shader = include_wgsl!("blit.wgsl");
        let depth_filter_pipeline = util::create_copy_quad_pipeline(
            device,
            &depth_filter_pipeline_layout,
            texture::FILTERED_DEPTH_FORMAT,
            None,
            &[],
            None,
            depth_filter_shader,
            Some("depth filter render"),
        );

        Self {
            ssao_commons,
            sampler,
            noise_sampler,
            depth_mip_sampler,
            hilbert_noise,
            depth_bind_group_ping,
            depth_bind_group_pong,
            depth_bind_group_layout,
            ssao_pipeline,
            frame_index,
            denoiser_bind_group_ping,
            denoiser_bind_group_pong,
            denoiser_bind_group_layout,
            denoiser_pipeline,
            cleared: false,
            depth_copy_bind_group,
            depth_copy_bind_group_layout,
            depth_copy_pipeline,
            depth_filter_bind_groups_ping,
            depth_filter_bind_groups_pong,
            depth_filter_bind_group_layout,
            depth_filter_pipeline,
            ping: true,
        }
    }

    pub fn render(
        &mut self,
        ssao_enabled: bool,
        mut encoder: &mut wgpu::CommandEncoder,
        #[cfg(feature = "profiling")] profiler: &GpuProfiler,
        ssao_view: &wgpu::TextureView,
        denoiser_edges_view: &wgpu::TextureView,
        denoised_ssao_view_ping: &wgpu::TextureView,
        denoised_ssao_view_pong: &wgpu::TextureView,
        filtered_depth_mip_views_ping: &[wgpu::TextureView;
             texture::FILTERED_DEPTH_MIP_LEVEL_COUNT as usize],
        filtered_depth_mip_views_pong: &[wgpu::TextureView;
             texture::FILTERED_DEPTH_MIP_LEVEL_COUNT as usize],
    ) -> bool {
        if ssao_enabled {
            self.cleared = false;
            self.frame_index += 1;
            #[cfg(feature = "profiling")]
            let mut encoder = profiler.scope("SSAO", encoder);
            let filtered_depth_mip_views = if self.ping {
                filtered_depth_mip_views_ping
            } else {
                filtered_depth_mip_views_pong
            };
            {
                let scope = cfg_select! {
                    feature = "profiling" => &mut encoder.scope("Copy & linearize depth"),
                    _ => &mut encoder,
                };
                let mut render_pass = scope.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Copy Depth Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &filtered_depth_mip_views[0],
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
                render_pass.set_bind_group(0, &self.depth_copy_bind_group, &[]);
                render_pass.set_pipeline(&self.depth_copy_pipeline);
                render_pass.draw(0..4, 0..1);
            }
            let depth_filter_bind_groups = if self.ping {
                &self.depth_filter_bind_groups_ping
            } else {
                &self.depth_filter_bind_groups_pong
            };
            {
                let scope = cfg_select! {
                    feature = "profiling" => &mut encoder.scope("Mip filter depth"),
                    _ => &mut encoder,
                };
                for (i, bind_group) in depth_filter_bind_groups.iter().enumerate() {
                    let mut render_pass = scope.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Mip filter Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &filtered_depth_mip_views[i + 1],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                        multiview_mask: None,
                    });
                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.set_pipeline(&self.depth_filter_pipeline);
                    render_pass.draw(0..4, 0..1);
                }
            }
            {
                let scope = cfg_select! {
                    feature = "profiling" => &mut encoder.scope("Main"),
                    _ => &mut encoder,
                };
                let mut render_pass = scope.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SSAO Render Pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: ssao_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: denoiser_edges_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                    ],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
                if self.ping {
                    render_pass.set_bind_group(0, &self.depth_bind_group_ping, &[]);
                } else {
                    render_pass.set_bind_group(0, &self.depth_bind_group_pong, &[]);
                };
                render_pass.set_pipeline(&self.ssao_pipeline);
                render_pass.draw(0..4, 0..1);
            }
            {
                let scope = cfg_select! {
                    feature = "profiling" => &mut encoder.scope("Denoiser"),
                    _ => &mut encoder,
                };
                let mut render_pass = scope.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SSAO Denoiser Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: if self.ping {
                            denoised_ssao_view_ping
                        } else {
                            denoised_ssao_view_pong
                        },
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
                });
                if self.ping {
                    render_pass.set_bind_group(0, &self.denoiser_bind_group_ping, &[]);
                } else {
                    render_pass.set_bind_group(0, &self.denoiser_bind_group_pong, &[]);
                }
                render_pass.set_pipeline(&self.denoiser_pipeline);
                render_pass.draw(0..4, 0..1);
            }
            self.ping = !self.ping;
            !self.ping
        } else if !self.cleared {
            self.ping = true;
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: denoised_ssao_view_ping,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.,
                            g: 1.,
                            b: 1.,
                            a: 1.,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            self.cleared = true;
            self.ping
        } else {
            self.ping
        }
    }

    pub fn update_reprojection(
        &mut self,
        queue: &wgpu::Queue,
        camera: &Camera,
        reproj: glam::Mat4,
        width: u32,
        height: u32,
    ) {
        let (depth_mul, depth_add) = camera.get_linearize_z_mul_add();
        let (x_mul, x_add) = camera.get_uv_to_view_x_mul_add();
        let (y_mul, y_add) = camera.get_uv_to_view_y_mul_add();
        let world_distance_inv = (1. + depth_add) / depth_mul;
        let pix_size_x = 1. / width as f32;
        let pix_size_y = 1. / height as f32;
        let params = SSAOParams {
            reprojection: reproj.to_cols_array_2d(),
            depth_mul,
            depth_add,
            x_mul,
            x_add,
            y_mul,
            y_add,
            frame_index: 289 * (self.frame_index % 64),
            world_distance_inv,
            pix_size: [pix_size_x, pix_size_y],
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.ssao_commons, 0, bytemuck::cast_slice(&[params]));
    }
}
