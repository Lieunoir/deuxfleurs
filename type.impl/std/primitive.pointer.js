(function() {
    var type_impls = Object.fromEntries([["ash",[]],["atk_sys",[]],["cairo_sys",[]],["gdk_pixbuf_sys",[]],["gdk_sys",[]],["gio_sys",[]],["glib_sys",[]],["gobject_sys",[]],["gtk_sys",[]],["khronos_egl",[]],["pango_sys",[]],["renderdoc_sys",[]],["wgpu_core",[]],["winit",[]],["x11_dl",[]]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()
//{"start":55,"fragment_lengths":[10,15,17,22,15,15,16,19,15,19,17,21,17,13,14]}