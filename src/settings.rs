use crate::data::Colors;
use egui::DragValue;
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};
use std::num::NonZeroU8;
use wgpu::Color;

/// Global rendering settings
#[derive(Clone)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "saves", serde(default))]
pub struct Settings {
    //vsync: bool,
    //show_fps: bool,
    /// Only redraw scene when window requires it
    pub lazy_draw: bool,
    /// Disable storing last redraw in buffer
    pub rerender: bool,
    /// Number of frame used in temporal anti aliasing, `None` disables taa
    ///
    /// TAA is crudely applied for a number of fixed frames when the scene stops changing
    pub taa: Option<NonZeroU8>,
    /// Ground shadow
    pub shadow: bool,
    /// Background color
    pub background_color: Color,
    pub mouse_sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub default_color_map: Colors,
    pub fit_camera_on_start: bool,
    pub ssao_enabled: bool,
    pub ssao_slice_per_pixel: u8,
    pub ssao_sample_per_slice: u8,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            lazy_draw: true,
            rerender: false,
            taa: NonZeroU8::new(16),
            shadow: true,
            background_color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            mouse_sensitivity: 0.1,
            zoom_sensitivity: 0.1,
            default_color_map: Colors::Viridis,
            fit_camera_on_start: true,
            ssao_enabled: true,
            ssao_slice_per_pixel: 3,
            ssao_sample_per_slice: 3,
        }
    }
}

impl Settings {
    pub fn draw_ui(&mut self, ui: &mut egui::Ui, refresh_screen: &mut bool) {
        ui.collapsing("Settings", |ui| {
            ui.horizontal(|ui| {
                let mut value = self.taa.map(|v| v.get()).unwrap_or(0) as f32;
                ui.add(
                    DragValue::new(&mut value)
                        .range(0..=64)
                        .prefix("TAA frames: "),
                );
                let value = NonZeroU8::new(value as u8);
                if self.taa != value {
                    self.taa = value;
                    *refresh_screen = true;
                }
            });
            *refresh_screen |= ui.checkbox(&mut self.shadow, "Ground shadow").clicked();
            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new(&mut self.mouse_sensitivity)
                        .range(0. ..=100.)
                        .speed(0.01)
                        .prefix("Mouse sensitivity: "),
                );
            });
            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new(&mut self.zoom_sensitivity)
                        .range(0. ..=100.)
                        .speed(0.01)
                        .prefix("Zoom sensitivity: "),
                );
            });
            if ui.checkbox(&mut self.ssao_enabled, "SSAO").changed() {
                *refresh_screen = true;
            }
            let mut value = self.ssao_slice_per_pixel as f32;
            ui.add(
                DragValue::new(&mut value)
                    .range(0..=16)
                    .prefix("Slice per pixel: "),
            );
            let value = value as u8;
            if self.ssao_slice_per_pixel != value {
                self.ssao_slice_per_pixel = value;
                *refresh_screen = true;
            }
            let mut value = self.ssao_sample_per_slice as f32;
            ui.add(
                DragValue::new(&mut value)
                    .range(0..=16)
                    .prefix("Sample per slice: "),
            );
            let value = value as u8;
            if self.ssao_sample_per_slice != value {
                self.ssao_sample_per_slice = value;
                *refresh_screen = true;
            }
        });
    }
}
