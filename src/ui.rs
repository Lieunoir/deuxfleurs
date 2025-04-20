use egui::style::{WidgetVisuals, Widgets};
use egui::Shadow;
use egui::{Color32, Rounding, Stroke};
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::State;
use indexmap::IndexMap;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub trait UiDataElement {
    fn draw(&mut self, ui: &mut egui::Ui, property_changed: &mut bool) -> bool;

    fn draw_gizmo(
        &mut self,
        _ui: &mut egui::Ui,
        _name: &str,
        _view: cgmath::Matrix4<f32>,
        _proj: cgmath::Matrix4<f32>,
        _gizmo_hovered: &mut bool,
    ) -> bool {
        false
    }
}

pub struct UI {
    rpass: Renderer,
    ctx: egui::Context,
    state: State,
    // used to pass the output at each frame without overloading the render loop
    // could be better
    platform_output: Option<egui::output::PlatformOutput>,
    pub(crate) hovered: bool,
}

fn blue_visuals() -> egui::Visuals {
    egui::Visuals {
        window_fill: egui::Color32::from_rgba_premultiplied(12, 0, 70, 220),
        window_stroke: egui::Stroke::NONE,
        extreme_bg_color: egui::Color32::from_rgba_premultiplied(10, 0, 50, 240),
        faint_bg_color: egui::Color32::from_rgba_premultiplied(10, 0, 50, 200),
        window_rounding: egui::Rounding::same(1.0),
        window_highlight_topmost: false,
        window_shadow: Shadow::NONE,
        widgets: Widgets {
            noninteractive: WidgetVisuals {
                weak_bg_fill: Color32::from_gray(27),
                bg_fill: Color32::from_gray(27),
                bg_stroke: Stroke::new(1.0, egui::Color32::from_black_alpha(200)), // separators, indentation lines
                fg_stroke: Stroke::new(1.0, Color32::from_gray(240)), // normal text color
                rounding: Rounding::same(2.0),
                expansion: 0.0,
            },
            inactive: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(160), // button background
                bg_fill: egui::Color32::from_black_alpha(160),      // checkbox background
                bg_stroke: Default::default(),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(240)), // button text
                rounding: Rounding::same(2.0),
                expansion: 0.0,
            },
            hovered: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(180), // button background
                bg_fill: egui::Color32::from_black_alpha(180),      // checkbox background
                bg_stroke: Stroke::new(1.0, Color32::from_gray(250)), // e.g. hover over window edge or button
                fg_stroke: Stroke::new(1.5, Color32::from_gray(250)),
                rounding: Rounding::same(3.0),
                expansion: 1.0,
            },
            active: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(150),
                bg_fill: egui::Color32::from_black_alpha(150),
                bg_stroke: Stroke::new(1.0, Color32::WHITE),
                fg_stroke: Stroke::new(2.0, Color32::WHITE),
                rounding: Rounding::same(2.0),
                expansion: 1.0,
            },
            open: WidgetVisuals {
                weak_bg_fill: Color32::from_gray(45),
                bg_fill: Color32::from_gray(27),
                bg_stroke: Stroke::new(1.0, Color32::from_gray(210)),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(210)),
                rounding: Rounding::same(2.0),
                expansion: 0.0,
            },
        },
        ..Default::default()
    }
}

fn transparent_visuals() -> egui::Visuals {
    egui::Visuals {
        window_fill: egui::Color32::from_black_alpha(220),
        window_stroke: egui::Stroke::NONE,
        extreme_bg_color: egui::Color32::from_black_alpha(220),
        faint_bg_color: egui::Color32::from_black_alpha(100),
        window_rounding: egui::Rounding::same(1.0),
        window_highlight_topmost: false,
        window_shadow: Shadow::NONE,
        widgets: Widgets {
            noninteractive: WidgetVisuals {
                weak_bg_fill: Color32::from_gray(27),
                bg_fill: Color32::from_gray(27),
                bg_stroke: Stroke::new(1.0, egui::Color32::from_black_alpha(80)), // separators, indentation lines
                fg_stroke: Stroke::new(1.0, Color32::from_gray(180)), // normal text color
                rounding: Rounding::same(2.0),
                expansion: 0.0,
            },
            inactive: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(100), // button background
                bg_fill: egui::Color32::from_black_alpha(100),      // checkbox background
                bg_stroke: Default::default(),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(200)), // button text
                rounding: Rounding::same(2.0),
                expansion: 0.0,
            },
            hovered: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(110), // button background
                bg_fill: egui::Color32::from_black_alpha(110),      // checkbox background
                bg_stroke: Stroke::new(1.0, Color32::from_gray(200)), // e.g. hover over window edge or button
                fg_stroke: Stroke::new(1.5, Color32::from_gray(220)),
                rounding: Rounding::same(3.0),
                expansion: 1.0,
            },
            active: WidgetVisuals {
                weak_bg_fill: egui::Color32::from_black_alpha(95),
                bg_fill: egui::Color32::from_black_alpha(95),
                bg_stroke: Stroke::new(1.0, Color32::WHITE),
                fg_stroke: Stroke::new(2.0, Color32::WHITE),
                rounding: Rounding::same(2.0),
                expansion: 1.0,
            },
            open: WidgetVisuals {
                weak_bg_fill: Color32::from_gray(45),
                bg_fill: Color32::from_gray(27),
                bg_stroke: Stroke::new(1.0, Color32::from_gray(60)),
                fg_stroke: Stroke::new(1.0, Color32::from_gray(210)),
                rounding: Rounding::same(2.0),
                expansion: 0.0,
            },
        },
        ..Default::default()
    }
}

impl UI {
    pub fn new(
        device: &wgpu::Device,
        event_loop: &ActiveEventLoop,
        target_format: wgpu::TextureFormat,
        scale_factor: f64,
    ) -> Self {
        let rpass = Renderer::new(device, target_format, None, 1, true);
        let ctx = egui::Context::default();
        //TODO some kind of styling
        let visuals = blue_visuals();
        ctx.set_visuals(visuals);
        /*
        let mut style: egui::Style = (*ctx.style()).clone();
        style.override_font_id = Some(egui::FontId::proportional(20.));
        ctx.set_style(style);
        */
        let state = State::new(
            ctx.clone(),
            egui::viewport::ViewportId::ROOT,
            event_loop,
            Some(scale_factor as f32),
            None,
            None,
        );
        Self {
            rpass,
            ctx,
            state,
            platform_output: None,
            hovered: false,
        }
    }

    pub fn process_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    pub fn draw_models(
        &mut self,
        window: &Window,
        surfaces: &mut IndexMap<String, crate::surface::Surface>,
        clouds: &mut IndexMap<String, crate::point_cloud::PointCloud>,
        curves: &mut IndexMap<String, crate::segment::Segment>,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
    ) {
        let input = self.state.take_egui_input(window);
        self.ctx.begin_pass(input);
        self.hovered = false;

        if let Some(response) = egui::Window::new("Models")
            .anchor(egui::Align2::LEFT_TOP, [5., 5.])
            .resizable(false)
            .vscroll(true)
            .default_width(270.)
            .min_height(650.)
            .show(&self.ctx, |ui| {
                for (name, surface) in surfaces.iter_mut() {
                    egui::CollapsingHeader::new(format!(
                        "{} : {} vertices, {} faces",
                        name,
                        surface.geometry().vertices.len(),
                        surface.geometry().indices.size(),
                    ))
                    .default_open(true)
                    .show(ui, |ui| {
                        surface.inner.draw_ui(ui);
                    });
                }

                for (name, cloud) in clouds.iter_mut() {
                    egui::CollapsingHeader::new(format!(
                        "{} : {} points",
                        name,
                        cloud.geometry().positions.len(),
                    ))
                    .default_open(true)
                    .show(ui, |ui| {
                        cloud.inner.draw_ui(ui);
                    });
                }

                for (name, curve) in curves.iter_mut() {
                    egui::CollapsingHeader::new(format!(
                        "{} : {} points, {} edges",
                        name,
                        curve.geometry().positions.len(),
                        curve.geometry().connections.len(),
                    ))
                    .default_open(true)
                    .show(ui, |ui| {
                        curve.inner.draw_ui(ui);
                    });
                }
                ui.allocate_space(ui.available_size());
            })
        {
            self.hovered |= response.response.contains_pointer();
        }
        egui::Area::new("Viewport".into())
            .fixed_pos((0.0, 0.0))
            .show(&self.ctx, |ui| {
                for (_, surface) in surfaces.iter_mut() {
                    surface.inner.draw_gizmo(ui, view, proj, &mut self.hovered);
                }
                for (_, curve) in curves.iter_mut() {
                    curve.inner.draw_gizmo(ui, view, proj, &mut self.hovered);
                }
                for (_, cloud) in clouds.iter_mut() {
                    cloud.inner.draw_gizmo(ui, view, proj, &mut self.hovered);
                }
            });
    }

    pub fn draw_callback<T: FnMut(&mut egui::Ui, &mut crate::State)>(
        &mut self,
        event_loop_proxy: &winit::event_loop::EventLoopProxy<crate::UserEvent>,
        state: &mut crate::State,
        callback: &mut T,
    ) {
        if let Some(response) = egui::Window::new("Interactions")
            .anchor(egui::Align2::RIGHT_TOP, [-5., 5.])
            .resizable(false)
            .show(&self.ctx, |ui| {
                if ui.add(egui::Button::new("Fit camera")).clicked() {
                    state.resize_scene();
                }
                if ui.add(egui::Button::new("Screenshot")).clicked() {
                    state.screenshot();
                }
                if ui.add(egui::Button::new("Load obj")).clicked() {
                    let event_loop_proxy = event_loop_proxy.clone();
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let file = rfd::FileDialog::new()
                            .add_filter("obj", &["obj"])
                            //.set_directory("/")
                            .pick_file();
                        if let Some(file_handle) = file {
                            let data = file_handle;
                            if let Ok((mesh_v, mesh_f)) =
                                crate::resources::load_mesh_blocking(data.into())
                            {
                                event_loop_proxy
                                    .send_event(crate::UserEvent::LoadMesh(mesh_v, mesh_f))
                                    .ok();
                            }
                        }
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        let file = rfd::AsyncFileDialog::new()
                            .add_filter("obj", &["obj"])
                            //.set_directory("/")
                            .pick_file();
                        let f = async move {
                            let file = file.await;
                            if let Some(file_handle) = file {
                                let data = file_handle.read().await;
                                if let Ok((mesh_v, mesh_f)) =
                                    crate::resources::parse_preloaded_mesh(data).await
                                {
                                    event_loop_proxy
                                        .send_event(crate::UserEvent::LoadMesh(mesh_v, mesh_f))
                                        .ok();
                                }
                            }
                        };
                        wasm_bindgen_futures::spawn_local(f);
                    }
                }
                if let Some((picked_name, picked_number)) = state.get_picked() {
                    let picked_number = *picked_number;
                    ui.label(format!("Picked {}", picked_name));
                    if let Some(surface) = state.get_surface(&picked_name) {
                        surface.draw_element_info(picked_number, ui);
                    } else if let Some(cloud) = state.get_point_cloud(&picked_name) {
                        cloud.draw_element_info(picked_number, ui);
                    } else if let Some(curve) = state.get_segment(&picked_name) {
                        curve.draw_element_info(picked_number, ui);
                    }
                }

                callback(ui, state)
            })
        {
            self.hovered |= response.response.contains_pointer()
        }
    }

    // not sure about returning a tuple here
    pub fn render_deltas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
    ) -> (
        Vec<wgpu::CommandBuffer>,
        Vec<egui::ClippedPrimitive>,
        ScreenDescriptor,
    ) {
        let full_output = self.ctx.end_pass();
        let textures_delta = full_output.textures_delta;
        let clipped_primitives = self
            .ctx
            .tessellate(full_output.shapes, self.ctx.pixels_per_point());

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: self.ctx.pixels_per_point(),
        };

        let user_cmd_bufs = {
            for (id, image_delta) in &textures_delta.set {
                self.rpass.update_texture(device, queue, *id, image_delta);
            }
            self.rpass.update_buffers(
                device,
                queue,
                encoder,
                &clipped_primitives,
                &screen_descriptor,
            )
        };
        self.platform_output = Some(full_output.platform_output);
        (user_cmd_bufs, clipped_primitives, screen_descriptor)
    }

    pub fn render(
        &self,
        mut render_pass: wgpu::RenderPass<'static>,
        clipped_primitives: &[egui::ClippedPrimitive],
        screen_descriptor: &ScreenDescriptor,
    ) {
        self.rpass
            .render(&mut render_pass, clipped_primitives, screen_descriptor);
    }

    pub fn handle_platform_output(&mut self, window: &Window) {
        if let Some(platform_output) = self.platform_output.take() {
            self.state.handle_platform_output(window, platform_output);
        }
    }
}
