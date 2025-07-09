fn main() {
    let mut handle = deuxfleurs::init();
    handle.load_from_state_file("examples/assets/deuxfleurs.cbor");
    handle.get_settings_mut().fit_camera_on_start = false;
    // Will display a snapshot taken from the `data_types` example.
    handle.run(1080, 720, Some("deuxfleurs"));
}
