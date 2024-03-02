use egui::Widget;
use egui_gizmo::GizmoMode;
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit::State;
use indexmap::IndexMap;
use winit::event_loop::EventLoop;
use winit::window::Window;

use crate::curve::CurveData;
use crate::point_cloud::CloudData;

pub trait UiDataElement {
    fn draw(&mut self, ui: &mut egui::Ui, property_changed: &mut bool) -> bool;

    fn draw_gizmo(&mut self,
        ui: &mut egui::Ui,
        name: &str,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
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
}

impl UI {
    pub fn new<T>(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        event_loop: &EventLoop<T>,
        scale_factor: f64,
    ) -> Self {
        let rpass = Renderer::new(
            device,
            target_format,
            None,
            1,
        );
        let ctx = egui::Context::default();
        //TODO some kind of styling
        let visuals = egui::Visuals {
            window_fill: egui::Color32::from_rgba_premultiplied(10, 0, 50, 192),
            ..Default::default()
        };
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
        );
        Self {
            rpass,
            ctx,
            state,
            platform_output: None,
        }
    }

    pub fn process_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    pub fn draw_models(
        &mut self,
        window: &Window,
        surfaces: &mut IndexMap<String, crate::surface::Surface>,
        clouds: &mut IndexMap<String, crate::point_cloud::PointCloud>,
        curves: &mut IndexMap<String, crate::curve::Curve>,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
    ) {
        let input = self.state.take_egui_input(window);
        self.ctx.begin_frame(input);

        egui::Window::new("Models")
            .anchor(egui::Align2::LEFT_TOP, [5., 5.])
            .resizable(false)
            .vscroll(true)
            .min_height(650.)
            .show(&self.ctx, |ui| {
                for (name, surface) in surfaces.iter_mut() {
                    egui::CollapsingHeader::new(format!(
                        "{} : {} vertices, {} faces",
                        name,
                        surface.geometry.vertices.len(),
                        surface.geometry.indices.size(),
                    ))
                    .default_open(true)
                    .show(ui, |ui| {
                        surface.draw_ui(ui);
                    });
                }

                for (name, cloud) in clouds.iter_mut() {
                    egui::CollapsingHeader::new(format!(
                        "{} : {} points",
                        name,
                        cloud.geometry.positions.len(),
                    ))
                    .default_open(true)
                    .show(ui, |ui| {
                        cloud.draw_ui(ui);
                    });
                }

                for (name, curve) in curves.iter_mut() {
                    egui::CollapsingHeader::new(format!(
                        "{} : {} points, {} edges",
                        name,
                        curve.geometry.positions.len(),
                        curve.geometry.connections.len(),
                    ))
                    .default_open(true)
                    .show(ui, |ui| {
                        curve.draw_ui(ui);
                    });
                }
                ui.allocate_space(ui.available_size());
            });
        egui::Area::new("Viewport")
            .fixed_pos((0.0, 0.0))
            .show(&self.ctx, |ui| {
                //Fix for not having a correct area
                //ui.with_layer_id(egui::LayerId::background(), |ui| {
                //ui.with_layer_id(egui::LayerId::background(), |ui| {
                //});
                    for (_, surface) in surfaces.iter_mut() {
                        surface.draw_gizmo(ui, view, proj);
                    }
                    for (_, curve) in curves.iter_mut() {
                        curve.draw_gizmo(ui, view, proj);
                    }
                    for (_, cloud) in clouds.iter_mut() {
                        cloud.draw_gizmo(ui, view, proj);
                    }
            });
    }

    pub fn draw_callback(
        &mut self,
        event_loop_proxy: &winit::event_loop::EventLoopProxy<crate::UserEvent>,
        state: &mut crate::State,
        callback: &mut Option<Box<dyn FnMut(&mut egui::Ui, &mut crate::State)>>,
    ) {
        egui::Window::new("Interactions")
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
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("obj", &["obj"])
                        //.set_directory("/")
                        .pick_file();
                    let event_loop_proxy = event_loop_proxy.clone();
                    let f = async move {
                        let file = file.await;
                        if let Some(file_handle) = file {
                            let data = file_handle.read().await;
                            if let Ok((mesh_v, mesh_f)) =
                                crate::resources::load_preloaded_mesh(data).await
                            {
                                event_loop_proxy
                                    .send_event(crate::UserEvent::LoadMesh(mesh_v, mesh_f))
                                    .ok();
                            }
                        }
                    };
                    //TODO avoid pollster blocking here
                    #[cfg(not(target_arch = "wasm32"))]
                    pollster::block_on(f);
                    #[cfg(target_arch = "wasm32")]
                    wasm_bindgen_futures::spawn_local(f);
                }
                if let Some((picked_name, picked_number)) = state.get_picked() {
                    let mut picked_number = *picked_number;
                    ui.label(format!("Picked {}", picked_name));
                    if let Some(surface) = state.get_surface(&picked_name) {
                        surface.draw_element_info(picked_number, ui);
                    } else if let Some(cloud) = state.get_point_cloud(&picked_name) {
                        cloud.draw_element_info(picked_number, ui);
                    } else if let Some(curve) = state.get_curve(&picked_name) {
                        curve.draw_element_info(picked_number, ui);
                    }
                }

                if let Some(callback) = callback {
                    callback(ui, state)
                }
            });
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
        let full_output = self.ctx.end_frame();
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

    pub fn render<'a, 'b: 'a>(
        &'b self,
        render_pass: &mut wgpu::RenderPass<'a>,
        clipped_primitives: &'a [egui::ClippedPrimitive],
        screen_descriptor: &'a ScreenDescriptor,
    ) {
        self.rpass
            .render(render_pass, clipped_primitives, screen_descriptor);
    }

    pub fn handle_platform_output(&mut self, window: &Window) {
        if let Some(platform_output) = self.platform_output.take() {
            self.state.handle_platform_output(window, platform_output);
        }
    }
}
