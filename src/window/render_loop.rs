use super::{InnerGraphicalState, JitterUniform, RunningState, StateWrapper, UserEvent};
use crate::camera::{Camera, CameraController, CameraUniform};
use crate::sbv::SBV;
#[cfg(feature = "saves")]
use crate::window::InnerBareStateSerde;
use crate::{Settings, post_process};

use crate::picker::{self, Picked};
use crate::point_cloud::UninitedPointCloud;
use crate::screenshot;
use crate::segment::UninitedSegment;
use crate::surface::UninitedSurface;
use crate::texture::TextureBufferPool;
#[cfg(not(target_arch = "wasm32"))]
use egui_winit::clipboard::Clipboard;
use indexmap::IndexMap;
use pollster::FutureExt;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::iter;
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::Clipboard;
use wgpu::rwh::HasDisplayHandle;
use wgpu::util::DeviceExt;
use wgpu::{CompositeAlphaMode, Extent3d};
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};
use winit::application::ApplicationHandler;
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey, SmolStr};
use winit::{
    dpi::PhysicalSize,
    event::*,
    window::{Window, WindowAttributes},
};

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(module = "/src/save.js")]
extern "C" {
    fn save_state(filename: &str, data: &[u8]);
}

impl InnerGraphicalState {
    // Initialize the state
    pub(crate) async fn new(
        surfaces: IndexMap<String, UninitedSurface>,
        clouds: IndexMap<String, UninitedPointCloud>,
        segments: IndexMap<String, UninitedSegment>,
        settings: Settings,
        camera: Camera,
        window: Option<Window>,
        proxy: Option<EventLoopProxy<UserEvent>>,
    ) -> Self {
        let size = match window.as_ref() {
            Some(window) => window.inner_size(),
            None => PhysicalSize {
                width: 1920,
                height: 1080,
            },
        };
        let window = window.map(Arc::new);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });
        let surface = window
            .as_ref()
            .map(|window| instance.create_surface(Arc::clone(&window)).unwrap());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: settings.power_preference,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Select a device to use
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                required_features: GpuProfiler::ALL_WGPU_TIMER_FEATURES,
                required_limits: if cfg!(target_arch = "wasm32") {
                    let mut limits = wgpu::Limits::downlevel_webgl2_defaults();
                    limits.max_texture_dimension_2d = adapter.limits().max_texture_dimension_2d;
                    limits.max_buffer_size = adapter.limits().max_buffer_size;
                    limits
                } else {
                    let mut limits = wgpu::Limits::default();
                    limits.max_buffer_size = adapter.limits().max_buffer_size;
                    limits
                },
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let profiler = GpuProfiler::new_with_tracy_client(
            GpuProfilerSettings::default(),
            adapter.get_info().backend,
            &device,
            &queue,
        )
        .unwrap_or_else(|err| match err {
            wgpu_profiler::CreationError::TracyClientNotRunning
            | wgpu_profiler::CreationError::TracyGpuContextCreationError(_) => {
                println!("Failed to connect to Tracy. Continuing without Tracy integration.");
                GpuProfiler::new(&device, GpuProfilerSettings::default())
                    .expect("Failed to create profiler")
            }
            _ => {
                panic!("Failed to create profiler: {err}");
            }
        });

        let (surface_format, alpha_mode) = match surface.as_ref() {
            Some(surface) => {
                let surface_caps = surface.get_capabilities(&adapter);
                (
                    surface_caps
                        .formats
                        .iter()
                        .copied()
                        .find(|f| f.is_srgb())
                        .unwrap_or(surface_caps.formats[0]),
                    surface_caps.alpha_modes[0],
                )
            }
            None => (
                wgpu::TextureFormat::Rgba8UnormSrgb,
                CompositeAlphaMode::Auto,
            ),
        };

        // Config for surface
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            //present_mode: surface_caps.present_modes[0],
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface
            .as_ref()
            .map(|surface| surface.configure(&device, &config));

        //window.request_inner_size(PhysicalSize::new(width, height));

        // Bind the camera to the shaders
        let mut new_camera = Camera::new(config.width as f32 / config.height as f32);
        new_camera.set_from_camera(camera);
        let camera = new_camera;
        let camera_controller = CameraController::new();

        let mut camera_uniform = CameraUniform::new();
        let sbv = SBV::default();
        camera_uniform.update_view_proj(&camera, &sbv, 0.);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let jitter_uniform = JitterUniform {
            x: 0.,
            y: 0.,
            _padding: [0; 2],
        };

        let jitter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Jitter Buffer"),
            contents: bytemuck::cast_slice(&[jitter_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create a bind group for camera buffer
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX
                            | wgpu::ShaderStages::FRAGMENT
                            | wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: jitter_buffer.as_entire_binding(),
                },
            ],
            label: Some("camera_bind_group"),
        });

        let screenshoter = screenshot::Screenshoter::new();

        let picker = picker::Picker::new(&device, size.width.max(1), size.height.max(1));

        let surfaces = surfaces
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.upgrade(
                        &device,
                        &camera_bind_group_layout,
                        &picker.bind_group_layout,
                    ),
                )
            })
            .collect();

        let clouds = clouds
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.upgrade(
                        &device,
                        &camera_bind_group_layout,
                        &picker.bind_group_layout,
                    ),
                )
            })
            .collect();
        let segments = segments
            .into_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.upgrade(
                        &device,
                        &camera_bind_group_layout,
                        &picker.bind_group_layout,
                    ),
                )
            })
            .collect();

        let texture_buffer_pool = TextureBufferPool::new(
            &device,
            wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            surface_format,
        );

        let copy = post_process::TextureCopy::new(
            &device,
            texture_buffer_pool.get_blend_stored_view(),
            texture_buffer_pool.get_blend_target_view(),
            surface_format,
        );
        let pbr_renderer = post_process::PBR::new(
            &device,
            &camera,
            surface_format,
            texture_buffer_pool.get_albedo_view(),
            texture_buffer_pool.get_normals_view(),
            texture_buffer_pool.get_denoised_ssao_view_ping(),
            texture_buffer_pool.get_denoised_ssao_view_pong(),
        );
        let ground = post_process::Ground::new(
            &device,
            surface_format,
            texture_buffer_pool.get_depth_view(),
            &camera_bind_group_layout,
            0.,
        );
        let ssao = post_process::SSAO::new(
            &device,
            &queue,
            texture_buffer_pool.get_normals_view(),
            texture_buffer_pool.get_ssao_view(),
            texture_buffer_pool.get_denoiser_edges_view(),
            texture_buffer_pool.get_denoised_ssao_view_ping(),
            texture_buffer_pool.get_denoised_ssao_view_pong(),
            texture_buffer_pool.get_depth_view(),
            texture_buffer_pool.get_filtered_depth_view_ping(),
            texture_buffer_pool.get_filtered_depth_mip_views_ping(),
            texture_buffer_pool.get_filtered_depth_view_pong(),
            texture_buffer_pool.get_filtered_depth_mip_views_pong(),
        );
        let should_resize = settings.fit_camera_on_start;
        InnerGraphicalState {
            surfaces,
            segments,
            clouds,
            settings,
            window,
            proxy,
            surface,
            device,
            queue,
            config,
            size,
            texture_buffer_pool,
            screenshoter,
            ctrl_pressed: false,
            camera,
            camera_controller,
            camera_buffer,
            camera_uniform,
            jitter_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            picker,
            //time: std::time::Instant::now(),
            dirty: true,
            egui_dirty: true,
            should_resize,
            copy,
            pbr_renderer,
            ground,
            ssao,
            taa_counter: 0,
            sbv: SBV::default(),
            rng: SmallRng::seed_from_u64(1),
            profiler,
        }
    }

    fn set_floor(&mut self) {
        use crate::shape::ShapeGeometry;
        let mut min_y = 0.;
        for surface in self.surfaces.values() {
            if surface.shown() {
                for p in surface.geometry().get_positions() {
                    let p = &surface
                        .transform
                        .get_transform()
                        .project_point3((*p).into());
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        for cloud in self.clouds.values() {
            if cloud.shown() {
                for p in cloud.geometry().get_positions() {
                    let p = &cloud.transform.get_transform().project_point3((*p).into());
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        for segment in self.segments.values() {
            if segment.shown() {
                for p in segment.geometry().get_positions() {
                    let p = &segment
                        .transform
                        .get_transform()
                        .project_point3((*p).into());
                    if p[1] < min_y {
                        min_y = p[1];
                    }
                }
            }
        }
        self.ground.set_level(&self.queue, min_y);
    }

    pub(crate) fn resize_scene(&mut self) {
        let mut size = None;
        let mut n = 0;
        let mut center = glam::Vec3::new(0., 0., 0.);
        for surface in self.surfaces.values() {
            if surface.shown() {
                let sbv = surface.sbv.transform(&surface.transform.get_transform());
                center += glam::Vec3::from_array(sbv.center);
                n += 1;
                if let Some(size) = &mut size {
                    if sbv.radius > *size {
                        *size = sbv.radius;
                    }
                } else {
                    size = Some(sbv.radius);
                }
            }
        }
        for cloud in self.clouds.values() {
            if cloud.shown() {
                let sbv = cloud.sbv.transform(&cloud.transform.get_transform());
                center += glam::Vec3::from_array(sbv.center);
                n += 1;
                if let Some(size) = &mut size {
                    if sbv.radius > *size {
                        *size = sbv.radius;
                    }
                } else {
                    size = Some(sbv.radius);
                }
            }
        }
        for segment in self.segments.values() {
            if segment.shown() {
                let sbv = segment.sbv.transform(&segment.transform.get_transform());
                center += glam::Vec3::from_array(sbv.center);
                n += 1;
                if let Some(size) = &mut size {
                    if sbv.radius > *size {
                        *size = sbv.radius;
                    }
                } else {
                    size = Some(sbv.radius);
                }
            }
        }
        if n > 0 {
            center = center / (n as f32);
        }
        self.dirty = true;
        self.camera
            .set_scene_size(size.unwrap_or_else(|| 1.), center);
        self.set_floor();
    }

    // Keeps state in sync with window size when changed
    pub(crate) fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface
                .as_mut()
                .map(|surface| surface.configure(&self.device, &self.config));
            self.texture_buffer_pool = TextureBufferPool::new(
                &self.device,
                Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                self.config.format,
            );
            self.camera.resize(new_size.width, new_size.height);
            self.picker.resize(new_size.width, new_size.height);
            self.copy.resize(
                &self.device,
                self.texture_buffer_pool.get_blend_stored_view(),
                self.texture_buffer_pool.get_blend_target_view(),
            );
            self.pbr_renderer.resize(
                &self.device,
                &self.camera,
                &self.queue,
                self.texture_buffer_pool.get_albedo_view(),
                self.texture_buffer_pool.get_normals_view(),
                self.texture_buffer_pool.get_denoised_ssao_view_ping(),
                self.texture_buffer_pool.get_denoised_ssao_view_pong(),
            );
            self.ssao.resize(
                &self.device,
                self.texture_buffer_pool.get_normals_view(),
                self.texture_buffer_pool.get_ssao_view(),
                self.texture_buffer_pool.get_denoiser_edges_view(),
                self.texture_buffer_pool.get_denoised_ssao_view_ping(),
                self.texture_buffer_pool.get_denoised_ssao_view_pong(),
                self.texture_buffer_pool.get_depth_view(),
                self.texture_buffer_pool.get_filtered_depth_view_ping(),
                self.texture_buffer_pool.get_filtered_depth_mip_views_ping(),
                self.texture_buffer_pool.get_filtered_depth_view_pong(),
                self.texture_buffer_pool.get_filtered_depth_mip_views_pong(),
            );
        }
    }

    // Handle input using WindowEvent
    pub(crate) fn input(&mut self, event: &WindowEvent, ui_hovered: bool) -> bool {
        // Send any input to camera controller
        let changed = self.camera_controller.process_events(event, ui_hovered);
        self.dirty |= changed;
        (!ui_hovered && self.picker.input(event)) || changed
    }

    pub(crate) fn update(&mut self) -> bool {
        // Sync local app state with camera
        let old_camera = self.camera.clone();
        self.camera_controller
            .update_camera(&mut self.camera, &self.settings);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.sbv, self.ground.level);
        let reprojection = self.camera.get_reprojection_from(&old_camera);
        self.ssao.update_reprojection(
            &self.queue,
            &self.camera,
            reprojection,
            self.config.width,
            self.config.height,
        );
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        let mut changed = self.dirty;

        let mut sbv = None;

        for surface in self.surfaces.values() {
            if surface.show {
                SBV::merge(
                    &mut sbv,
                    &surface.sbv.transform(&surface.transform.get_transform()),
                );
            }
        }

        for cloud in self.clouds.values() {
            if cloud.show {
                SBV::merge(
                    &mut sbv,
                    &cloud.sbv.transform(&cloud.transform.get_transform()),
                );
            }
        }

        for segment in self.segments.values() {
            if segment.show {
                SBV::merge(
                    &mut sbv,
                    &segment.sbv.transform(&segment.transform.get_transform()),
                );
            }
        }

        self.sbv = sbv.unwrap_or_else(SBV::default);
        if self.should_resize {
            self.resize_scene();
            self.should_resize = false;
            changed = true;
        }

        self.dirty = false;
        changed
    }

    // Primary render flow
    pub(crate) fn render(
        &mut self,
        event_loop_proxy: Option<&winit::event_loop::EventLoopProxy<crate::window::UserEvent>>,
        mut ui: Option<&mut crate::ui::UI>,
        scene_changed: bool,
    ) -> Result<bool, wgpu::SurfaceError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        let deltas = ui.as_mut().map(|ui| {
            let deltas = ui.render_deltas(
                &self.device,
                &self.queue,
                &mut encoder,
                self.size.width,
                self.size.height,
            );

            if self.picker.render(
                &self.device,
                &mut encoder,
                &self.texture_buffer_pool.get_picker_view(),
                &self.texture_buffer_pool.get_depth_view(),
                &self.camera_bind_group,
                &self.surfaces,
                &self.clouds,
                &self.segments,
            ) {
                self.egui_dirty = true;
                self.texture_buffer_pool
                    .copy_picker_texture_to_buffer(&mut encoder);
            }
            deltas
        });

        // Hack for screenshot frames
        let output = if event_loop_proxy.is_some() {
            self.surface
                .as_ref()
                .map(|surface| surface.get_current_texture())
                .transpose()?
        } else {
            None
        };

        let view = output.as_ref().map(|o| {
            o.texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        });

        let mut request_redraw = !self.settings.lazy_draw;
        // Render, as opposed as simply getting the last stored frame
        let mut render = self.settings.rerender;
        //let mut store_render = self.surface.is_none();
        let mut store_render = event_loop_proxy.is_none();
        // ^ Three possibilities:
        // *  (true, false): continuously rendering
        // *  (true, true): scene hasn't changed, rendering more for TAA
        // *  (false, false): scene hasn't changed, just copy the last stored frame
        let jitter;
        if scene_changed || self.settings.rerender {
            // We rerender the scene from scratch
            request_redraw |= scene_changed;
            render = true;
            self.taa_counter = 0;
            jitter = JitterUniform {
                x: 0.,
                y: 0.,
                _padding: [0; 2],
            };
        } else {
            if let Some(taa_frames) = self.settings.taa {
                if self.taa_counter < taa_frames.get() {
                    // The scene hasn't changed but we need more copies for taa
                    render = true;
                    store_render = true;
                    request_redraw = true;
                    self.taa_counter += 1;
                }
                let ampli = 1.;
                jitter = JitterUniform {
                    x: ampli * 2. * (self.rng.random::<f32>() - 0.5) / self.size.width as f32,
                    y: ampli * 2. * (self.rng.random::<f32>() - 0.5) / self.size.height as f32,
                    _padding: [0; 2],
                };
            } else {
                if self.taa_counter == 0 {
                    render = true;
                    store_render = true;
                    self.taa_counter = 1;
                }
                jitter = JitterUniform {
                    x: 0.,
                    y: 0.,
                    _padding: [0; 2],
                };
            }
        };
        self.queue
            .write_buffer(&self.jitter_buffer, 0, bytemuck::cast_slice(&[jitter]));

        if render {
            let (view_ref, color) = if store_render {
                (
                    self.texture_buffer_pool.get_blend_target_view(),
                    wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    },
                )
            } else {
                (view.as_ref().unwrap(), self.settings.background_color)
            };

            let mut scope = self.profiler.scope("Material", &mut encoder);

            let mut material_render_pass = scope.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Material Render Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: self.texture_buffer_pool.get_albedo_view(),
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
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: self.texture_buffer_pool.get_normals_view(),
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
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.texture_buffer_pool.get_depth_view(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            material_render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

            //order matters!
            //cloud discard so no depth test
            //segment only change deph sometimes, could use conservative depth
            //surface fully uses depth buffer(except attachments)
            for cloud in self.clouds.values() {
                cloud.render(&mut material_render_pass);
            }
            for segment in self.segments.values() {
                segment.render(&mut material_render_pass);
            }
            for surface in self.surfaces.values() {
                surface.render(&mut material_render_pass);
            }
            drop(material_render_pass);
            drop(scope);

            let ping = self.ssao.render(
                self.settings.ssao_enabled,
                &mut encoder,
                &self.profiler,
                self.texture_buffer_pool.get_ssao_view(),
                self.texture_buffer_pool.get_denoiser_edges_view(),
                self.texture_buffer_pool.get_denoised_ssao_view_ping(),
                self.texture_buffer_pool.get_denoised_ssao_view_pong(),
                self.texture_buffer_pool.get_filtered_depth_mip_views_ping(),
                self.texture_buffer_pool.get_filtered_depth_mip_views_pong(),
            );
            let mut scope = self.profiler.scope("PBR", &mut encoder);
            let mut pbr_render_pass = scope.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PBR Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view_ref,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.pbr_renderer.render(&mut pbr_render_pass, ping);
            drop(pbr_render_pass);
            drop(scope);
            if self.settings.shadow {
                let mut scope = self.profiler.scope("Shadow", &mut encoder);
                let mut shadow_render_pass = scope.scoped_render_pass(
                    "Render shadow",
                    wgpu::RenderPassDescriptor {
                        label: Some("Shadow Render Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: self.ground.get_texture_view(),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.,
                                    g: 0.,
                                    b: 0.,
                                    a: 0.,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                    },
                );
                shadow_render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                for surface in self.surfaces.values() {
                    surface.render_shadow(&mut shadow_render_pass);
                }

                drop(shadow_render_pass);
                self.ground.blur(&mut scope);
                let mut ground_render_pass = scope.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Shadow Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: view_ref,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.texture_buffer_pool.get_depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                ground_render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                self.ground.render(&mut ground_render_pass);
                drop(ground_render_pass);
            }

            //Blend with previous frame
            if store_render {
                let factor = (self.taa_counter as f64 - 1.) / (self.taa_counter as f64);
                self.copy.blend(
                    &mut encoder,
                    factor,
                    self.taa_counter == 1,
                    self.texture_buffer_pool.get_blend_stored_view(),
                );
            }
        }

        //If we have a surface to display to, forward result to surface
        if let Some(view_ref) = view.as_ref()
            // Isn't needed for direct rendering, is for all others
            && !(render && !store_render)
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Copy Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_ref,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.settings.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.copy.copy(&mut render_pass);
        }

        if let Some(ui) = ui.as_mut()
            && let Some(view_ref) = view.as_ref()
            && let Some((_user_cmd_bufs, clipped_primitives, screen_descriptor)) = deltas.as_ref()
        {
            let ui_render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Ui Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: view_ref,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    // Create a depth stencil buffer using the depth texture
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                })
                .forget_lifetime();
            ui.render(ui_render_pass, &clipped_primitives, &screen_descriptor);
        }

        self.profiler.resolve_queries(&mut encoder);

        if let Some((user_cmd_bufs, _clipped_primitives, _screen_descriptor)) = deltas {
            self.queue.submit(
                user_cmd_bufs
                    .into_iter()
                    .chain(iter::once(encoder.finish())),
            );
            output.unwrap().present();
        } else {
            self.queue.submit(iter::once(encoder.finish()));
        }
        self.profiler.end_frame().unwrap();
        self.profiler
            .process_finished_frame(self.queue.get_timestamp_period());

        event_loop_proxy.map(|proxy| {
            self.picker.post_render(
                proxy,
                self.texture_buffer_pool.get_output_buffer(),
                self.texture_buffer_pool.get_output_buffer_dimensions(),
            );
        });
        if self.picker.pick_locked || !self.settings.lazy_draw {
            request_redraw = true;
        }
        Ok(request_redraw)
    }

    fn render_screenshot(&mut self) -> wgpu::SubmissionIndex {
        // When in headless mode, or when continuously updating,
        // we have to make sure an image can be copied from
        if self.surface.is_none() || self.taa_counter == 0 {
            self.update();
            for i in 0..self.settings.minimum_frame_per_screenshot.get() {
                self.render(None, None, i == 0).unwrap();
            }
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Screenshot Render Encoder"),
            });
        self.texture_buffer_pool
            .copy_screenshot_texture_to_buffer(&mut encoder);
        self.queue.submit(iter::once(encoder.finish()))
    }

    pub(crate) fn screenshot_to_buffer(&mut self) -> Result<Vec<u8>, ()> {
        let index = self.render_screenshot();
        self.screenshoter.create_image_buffer(
            &self.device,
            index,
            self.texture_buffer_pool.get_output_buffer(),
            self.texture_buffer_pool.get_output_buffer_dimensions(),
        )
    }

    pub(crate) fn screenshot(&mut self) {
        let index = self.render_screenshot();
        self.screenshoter.create_png(
            &self.device,
            index,
            self.texture_buffer_pool.get_output_buffer(),
            self.texture_buffer_pool.get_output_buffer_dimensions(),
        );
    }

    pub(crate) fn get_picked(&self) -> &Option<(String, Picked)> {
        &self.picker.picked_item
    }

    pub(crate) fn refresh(&mut self) {
        self.dirty = true;
    }

    #[cfg(feature = "surface_button")]
    pub(crate) fn send_mesh(&mut self, name: String) {
        let event_loop_proxy = self.proxy.clone();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = rfd::FileDialog::new()
                .set_parent(&*self.window.as_ref().unwrap())
                .add_filter("obj, off", &["obj", "off"])
                .pick_file();
            if let Some(file_handle) = file {
                let data = file_handle;
                if let Some((mesh_v, mesh_f)) = crate::resources::load_mesh_blocking(data) {
                    event_loop_proxy
                        .unwrap()
                        .send_event(UserEvent::LoadMesh(mesh_v, mesh_f, name))
                        .ok();
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let file = rfd::AsyncFileDialog::new()
                .add_filter("obj, off", &["obj", "off"])
                .pick_file();
            let f = async move {
                let file = file.await;
                if let Some(file_handle) = file {
                    let data = file_handle.read().await;
                    if let Some((mesh_v, mesh_f)) =
                        crate::resources::parse_preloaded_mesh(file_handle.file_name(), data).await
                    {
                        event_loop_proxy
                            .unwrap()
                            .send_event(UserEvent::LoadMesh(mesh_v, mesh_f, name))
                            .ok();
                    }
                }
            };
            wasm_bindgen_futures::spawn_local(f);
        }
    }

    #[cfg(feature = "saves")]
    pub(crate) fn save(&self) {
        let surfaces = self
            .surfaces
            .iter()
            .map(|(name, field)| (name.clone(), field.downgrade()))
            .collect();
        let clouds = self
            .clouds
            .iter()
            .map(|(name, field)| (name.clone(), field.downgrade()))
            .collect();
        let segments = self
            .segments
            .iter()
            .map(|(name, field)| (name.clone(), field.downgrade()))
            .collect();
        let bared = InnerBareStateSerde {
            settings: self.settings.clone(),
            camera: self.camera.clone(),
            surfaces,
            clouds,
            segments,
            ground_level: self.ground.level,
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(pathbuf) = rfd::FileDialog::new()
                .set_file_name("deuxfleurs.cbor")
                .set_parent(&*self.window.as_ref().unwrap())
                .save_file()
            {
                if let Ok(file) = std::fs::File::options()
                    .write(true)
                    .create(true)
                    .open(pathbuf)
                {
                    let buf_writer = std::io::BufWriter::new(file);
                    let _ = serde_cbor::to_writer(buf_writer, &bared);
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Ok(blob) = serde_cbor::to_vec(&bared) {
                save_state("deuxfleurs.cbor", &blob);
            }
        }
    }

    #[cfg(feature = "saves")]
    pub(crate) fn receive_save(&mut self, bared: InnerBareStateSerde) {
        self.surfaces = bared
            .surfaces
            .into_iter()
            .map(|(name, field)| {
                (
                    name,
                    field.upgrade(
                        &self.device,
                        &self.camera_bind_group_layout,
                        &self.picker.bind_group_layout,
                    ),
                )
            })
            .collect();
        self.clouds = bared
            .clouds
            .into_iter()
            .map(|(name, field)| {
                (
                    name,
                    field.upgrade(
                        &self.device,
                        &self.camera_bind_group_layout,
                        &self.picker.bind_group_layout,
                    ),
                )
            })
            .collect();
        self.segments = bared
            .segments
            .into_iter()
            .map(|(name, field)| {
                (
                    name,
                    field.upgrade(
                        &self.device,
                        &self.camera_bind_group_layout,
                        &self.picker.bind_group_layout,
                    ),
                )
            })
            .collect();
        self.camera.set_from_camera(bared.camera);
        self.settings = bared.settings;
        self.ground.set_level(&mut self.queue, bared.ground_level);
        self.dirty = true;
    }

    #[cfg(feature = "saves")]
    pub(crate) fn load(&mut self) {
        let event_loop_proxy = self.proxy.clone();
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(pathbuf) = rfd::FileDialog::new()
                .set_file_name("deuxfleurs.cbor")
                .add_filter("cbor", &["cbor"])
                .set_parent(&*self.window.as_ref().unwrap())
                .pick_file()
            {
                if let Ok(file) = std::fs::File::options().read(true).open(pathbuf) {
                    let buf_reader = std::io::BufReader::new(file);
                    if let Ok(bared) = serde_cbor::from_reader(buf_reader) {
                        event_loop_proxy
                            .unwrap()
                            .send_event(UserEvent::LoadState(bared))
                            .ok();
                    }
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let file = rfd::AsyncFileDialog::new()
                .set_file_name("deuxfleurs.cbor")
                .add_filter("cbor", &["cbor"])
                .pick_file();
            let f = async move {
                let file = file.await;
                if let Some(file_handle) = file {
                    let data = file_handle.read().await;
                    if let Ok(bared) = serde_cbor::from_slice(&data) {
                        event_loop_proxy
                            .unwrap()
                            .send_event(UserEvent::LoadState(bared))
                            .ok();
                    }
                }
            };
            wasm_bindgen_futures::spawn_local(f);
        }
    }
}

impl<T: FnMut(&mut egui::Ui, &mut RunningState)> ApplicationHandler<UserEvent> for StateWrapper<T> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let window_attributes = WindowAttributes::default()
                .with_title(&self.id)
                .with_inner_size(PhysicalSize::new(self.width, self.height));
            let window = event_loop.create_window(window_attributes).unwrap();

            #[cfg(target_arch = "wasm32")]
            {
                use winit::platform::web::WindowExtWebSys;
                web_sys::window()
                    .and_then(|win| {
                        self.clipboard = Some(win.navigator().clipboard());
                        win.document()
                    })
                    .and_then(|doc| {
                        let dst = doc.get_element_by_id(&self.id)?;
                        let canvas = window.canvas()?;
                        // disable right click
                        let empty_func = js_sys::Function::new_no_args("return false;");
                        canvas.set_oncontextmenu(Some(&empty_func));
                        dst.append_child(&canvas).ok()?;
                        Some(())
                    })
                    .expect("Couldn't append canvas to document body.");
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.clipboard = Some(Clipboard::new(
                    window.display_handle().ok().map(|handle| handle.as_raw()),
                ));
            }

            let init = self.init_state.take().unwrap();
            self.state = Some(
                RunningState::new(
                    init.0.surfaces,
                    init.0.clouds,
                    init.0.segments,
                    init.0.camera,
                    init.0.settings,
                    window,
                    self.proxy.clone(),
                )
                .block_on(),
            );
            self.ui = Some(crate::ui::UI::new(
                &self.state.as_ref().unwrap().device,
                event_loop,
                self.state.as_ref().unwrap().config.format,
                self.state
                    .as_ref()
                    .unwrap()
                    .window
                    .as_ref()
                    .unwrap()
                    .scale_factor(),
            ));
        }
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        if let Some(state) = self.state.as_mut() {
            match event {
                #[cfg(feature = "surface_button")]
                UserEvent::LoadMesh(mesh_v, mesh_f, name) => {
                    state.register_surface(name, mesh_v, mesh_f);
                }
                #[cfg(feature = "saves")]
                UserEvent::LoadState(bared) => {
                    state.receive_save(bared);
                }
                UserEvent::Paste(cam) => {
                    state.camera.set(cam);
                    state.dirty = true;
                }
                UserEvent::Pick => {
                    state.0.picker.pick(
                        &state.0.surfaces,
                        &state.0.clouds,
                        &state.0.segments,
                        &state.0.camera,
                        state.0.texture_buffer_pool.get_output_buffer(),
                        state.0.texture_buffer_pool.get_output_buffer_dimensions(),
                    );
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        //TODO the `if let` stuff is hacky and messy
        let (processed, ui_hovered) =
            if let (Some(ui), Some(state)) = (self.ui.as_mut(), self.state.as_mut()) {
                if window_id == state.window.as_ref().unwrap().id() {
                    (
                        ui.process_event(&*state.window.as_ref().unwrap(), &event),
                        ui.hovered,
                    )
                } else {
                    return;
                }
            } else {
                return;
            };

        let input = if let Some(state) = self.state.as_mut() {
            state.input(&event, ui_hovered)
        } else {
            false
        };

        if processed.repaint {
            // Attempt to make egui finish animations
            if let Some(state) = self.state.as_mut() {
                match event {
                    WindowEvent::RedrawRequested => {
                        if state.egui_dirty {
                            state.window.as_ref().unwrap().request_redraw();
                            state.egui_dirty = false;
                        } else {
                            state.egui_dirty = true;
                        }
                    }
                    _ => state.window.as_ref().unwrap().request_redraw(),
                }
            }
        }
        if !processed.consumed && !input {
            // Handle window events (like resizing, or key inputs)
            // This is stuff from `winit` -- see their docs for more info
            if let (Some(state), Some(ui)) = (self.state.as_mut(), self.ui.as_mut()) {
                match event {
                    WindowEvent::CloseRequested => event_loop.exit(),
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size);
                        state.dirty = true;
                        state.window.as_ref().unwrap().request_redraw();
                    }
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key,
                                //state: ElementState::Pressed,
                                state: key_state,
                                ..
                            },
                        ..
                    } => {
                        if logical_key == Key::Named(NamedKey::Control) {
                            if key_state == ElementState::Pressed {
                                state.ctrl_pressed = true;
                            } else if key_state == ElementState::Released {
                                state.ctrl_pressed = false;
                            }
                        }
                        if state.ctrl_pressed
                            && logical_key == Key::Character(SmolStr::new_inline("c"))
                            && key_state == ElementState::Pressed
                        {
                            if let Ok(cam) = state.camera.copy() {
                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    let clipboard = self.clipboard.as_mut().unwrap();
                                    clipboard.set_text(cam);
                                }
                                #[cfg(target_arch = "wasm32")]
                                {
                                    let promise = self.clipboard.as_mut().unwrap().write_text(&cam);
                                    let f = async move {
                                        wasm_bindgen_futures::JsFuture::from(promise).await;
                                    };
                                    wasm_bindgen_futures::spawn_local(f);
                                }
                            }
                        } else if state.ctrl_pressed
                            && logical_key == Key::Character(SmolStr::new_inline("v"))
                            && key_state == ElementState::Pressed
                        {
                            let clipboard = self.clipboard.as_mut().unwrap();
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                if let Some(cam) = clipboard.get() {
                                    state
                                        .proxy
                                        .as_mut()
                                        .unwrap()
                                        .send_event(crate::window::UserEvent::Paste(cam))
                                        .ok();
                                    //state.camera.set(cam);
                                    //state.dirty = true;
                                };
                            }
                            #[cfg(target_arch = "wasm32")]
                            {
                                let promise = self.clipboard.as_mut().unwrap().read_text();
                                let event_loop_proxy = state.proxy.clone();
                                let f = async move {
                                    if let Ok(res) =
                                        wasm_bindgen_futures::JsFuture::from(promise).await
                                    {
                                        if let Some(cam) = res.as_string() {
                                            event_loop_proxy
                                                .unwrap()
                                                .send_event(UserEvent::Paste(cam))
                                                .ok();
                                        }
                                    }
                                };
                                wasm_bindgen_futures::spawn_local(f);
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        //draw ui
                        //let mut refresh_screen = state.state.dirty;
                        ui.draw_models(
                            &*state.0.window.as_ref().unwrap(),
                            &mut state.0.surfaces,
                            &mut state.0.clouds,
                            &mut state.0.segments,
                            &mut state.0.picker,
                            state.0.camera.build_view(),
                            state.0.camera.build_proj(),
                            &state.0.device,
                            &state.0.queue,
                            &state.0.camera_bind_group_layout,
                            &mut state.0.dirty,
                        );
                        ui.draw_callback(state, &mut self.callback);
                        let scene_changed = state.update();
                        //actual rendering
                        match state.render(Some(&self.proxy), Some(ui), scene_changed) {
                            Ok(request_redraw) => {
                                if request_redraw {
                                    state.window.as_ref().unwrap().request_redraw();
                                }
                                ui.handle_platform_output(&*state.window.as_ref().unwrap())}
                            ,
                            // Reconfigure the surface if it's lost or outdated
                            Err(
                                wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                            ) => {
                                state.0.resize(state.0.size)
                            },
                            // The system is out of memory, we should probably quit
                            Err(wgpu::SurfaceError::OutOfMemory) | Err(wgpu::SurfaceError::Other) => event_loop.exit(),

                            Err(wgpu::SurfaceError::Timeout) => {
                                log::warn!("Surface timeout")
                            }
                        }
                    }
                    _ => {}
                }
            }
        } else {
            if let Some(state) = self.state.as_mut() {
                state.window.as_ref().unwrap().request_redraw();
            }
        }
    }
}
