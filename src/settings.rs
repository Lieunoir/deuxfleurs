use wgpu::Color;

#[derive(Clone)]
pub struct Settings {
    //vsync: bool,
    //show_fps: bool,
    pub continuous_refresh: bool,
    pub rerender: bool,
    pub taa: bool,
    pub taa_frames: u32,
    pub shadow: bool,
    pub color: Color,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            continuous_refresh: false,
            rerender: false,
            taa: true,
            taa_frames: 16,
            shadow: true,
            color: Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 0.0,
            },
        }
    }
}

impl Settings {
    pub fn ui(&mut self, _ui: &mut egui::Ui) {}
}
