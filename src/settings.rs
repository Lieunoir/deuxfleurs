pub struct Settings {
    //vsync: bool,
    //show_fps: bool,
    pub continuous_refresh: bool,
    pub rerender: bool,
    pub taa: bool,
    pub taa_frames: u32,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            continuous_refresh: false,
            rerender: false,
            taa: true,
            taa_frames: 16,
        }
    }
}

impl Settings {
    pub fn ui(&mut self, _ui: &mut egui::Ui) {}
}
