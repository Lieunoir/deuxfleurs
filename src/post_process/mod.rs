use crate::texture;
use crate::util;
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

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
            bind_group_layouts: &[&copy_bind_group_layout],
            push_constant_ranges: &[],
        });
        let blend_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blend Pipeline Layout"),
                bind_group_layouts: &[&copy_bind_group_layout],
                push_constant_ranges: &[],
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
    sampler: wgpu::Sampler,
    material_bind_group: wgpu::BindGroup,
    material_bind_group_layout: wgpu::BindGroupLayout,
    pbr_pipeline: wgpu::RenderPipeline,
}

impl PBR {
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        albedo_view: &wgpu::TextureView,
        normals_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        self.material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        albedo_view: &wgpu::TextureView,
        normals_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
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
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
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
