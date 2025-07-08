use deuxfleurs::{Settings, load_mesh};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

fn main() {
    pollster::block_on(run());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let (spot_v, spot_f) = load_mesh("examples/assets/spot.obj").await.unwrap();
    let mut handle = deuxfleurs::init(Settings::default());
    handle.register_surface("Spot", spot_v, spot_f);
    let mut handle = handle.run_headless();
    handle.screenshot();
}
