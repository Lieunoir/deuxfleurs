use crate::ui::UiDataElement;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct ColorSettings {
    pub color: [f32; 4],
}

fn hash(string: &str) -> u8 {
    let mut res = 0_u32;
    for char in string.as_bytes() {
        res = (res.overflowing_shl(5).0.overflowing_sub(res).0)
            .overflowing_add(*char as u32)
            .0;
    }
    (res % 255) as u8
}

pub fn oklch_to_linear(l: f32, c: f32, h: f32, alpha: f32) -> [f32; 4] {
    let a = c * h.cos();
    let b = c * h.sin();

    let l_ = l + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = l - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = l - 0.0894841775 * a - 1.2914855480 * b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    [
        4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        alpha,
    ]
}

impl ColorSettings {
    pub fn new(name: &str) -> Self {
        let hash = hash(name);
        Self {
            color: oklch_to_linear(0.6459, 0.1413, (hash as f32) / 256. * 2. * PI, 1.),
        }
    }
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            color: [0.2, 0.2, 0.8, 1.],
        }
    }
}

impl UiDataElement for ColorSettings {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut mesh_color = egui::Rgba::from_rgba_unmultiplied(
            self.color[0],
            self.color[1],
            self.color[2],
            self.color[3],
        );
        let mut changed = false;
        ui.horizontal(|ui| {
            changed = egui::widgets::color_picker::color_edit_button_rgba(
                ui,
                &mut mesh_color,
                egui::widgets::color_picker::Alpha::Opaque,
            )
            .changed();
        });
        self.color = mesh_color.to_array();
        changed
    }
}
