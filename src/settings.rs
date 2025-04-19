use std::num::NonZeroU8;

use wgpu::Color;

/// Global rendering settings
#[derive(Clone)]
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
    pub color: Color,
    pub mouse_sensitivity: f32,
    pub zoom_sensitivity: f32,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            lazy_draw: true,
            rerender: false,
            taa: NonZeroU8::new(16),
            shadow: true,
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 0.0,
            },
            mouse_sensitivity: 0.1,
            zoom_sensitivity: 0.1,
        }
    }
}

impl Settings {
    pub fn ui(&mut self, _ui: &mut egui::Ui) {}
}
