use cgmath::prelude::*;
use winit::event::*;

// Camera informations for easier updating
pub struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            eye: (0.0, 0.0, -5.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }

    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        proj * view
    }

    pub fn build_view(&self) -> cgmath::Matrix4<f32> {
        cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn build_proj(&self) -> cgmath::Matrix4<f32> {
        cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn get_position(&self) -> [f32; 3] {
        self.eye.into()
    }
}

// Camera matrices for GPU computations
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        // We're using Vector4 because ofthe camera_uniform 16 byte spacing requirement
        self.view_position = camera.eye.to_homogeneous().into();
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

// TODO : abstract this into a trait implemented by different controllers (arcball, fps...)
pub struct CameraController {
    /*
    speed: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    */
    is_mouse_left_pressed: bool,
    is_mouse_right_pressed: bool,
    prev_mouse: Option<winit::dpi::PhysicalPosition<f64>>,
    wheel_delta: Option<f32>,
    pan_delta: Option<(f32, f32)>,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            /*
            speed,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            */
            is_mouse_left_pressed: false,
            is_mouse_right_pressed: false,
            prev_mouse: None,
            wheel_delta: None,
            pan_delta: None,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } if self.prev_mouse.is_none() => {
                self.prev_mouse = Some(*position);
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let prev = self.prev_mouse.unwrap();
                self.pan_delta = Some(((position.x - prev.x) as f32, (position.y - prev.y) as f32));
                self.prev_mouse = Some(*position);
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Left {
                    self.is_mouse_left_pressed = *state == ElementState::Pressed;
                } else if *button == MouseButton::Right {
                    self.is_mouse_right_pressed = *state == ElementState::Pressed;
                }
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.wheel_delta = Some(match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => 0.002 * p.y as f32,
                });
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        /*
        let forward_mag = forward.magnitude();

        // Prevents glitching when camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        */
        if let Some(delta) = self.wheel_delta {
            camera.eye += forward_norm * delta;
            camera.target += forward_norm * delta;
            self.wheel_delta = None;
        }
        /*
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }
        */

        let right = forward_norm.cross(camera.up);
        if let Some((dx, dy)) = self.pan_delta {
            if self.is_mouse_left_pressed {
                let origin = cgmath::Point3::<f32> {
                    x: 0.,
                    y: 0.,
                    z: 0.,
                };
                let eye_norm = (camera.eye - cgmath::point3::<f32>(0., 0., 0.)).normalize();
                let eye_mag = (camera.eye - cgmath::point3::<f32>(0., 0., 0.)).magnitude();
                let center_right = eye_norm.cross(camera.up);
                let old_eye = camera.eye;
                camera.eye = cgmath::point3::<f32>(0., 0., 0.)
                    + (eye_norm + center_right * dx * 0.01 + camera.up * dy * 0.01).normalize()
                        * eye_mag;
                let rotation =
                    cgmath::Quaternion::between_vectors(old_eye - origin, camera.eye - origin);
                camera.target = origin + rotation.rotate_vector(camera.target - origin);
                camera.up =
                    rotation.rotate_vector(old_eye + camera.up - origin) - (camera.eye - origin);
                //camera.target = camera.eye + forward;
            } else if self.is_mouse_right_pressed {
                camera.eye += 0.03 * (-dx * right + dy * camera.up);
                camera.target += 0.03 * (-dx * right + dy * camera.up);
            }
            self.pan_delta = None;
        }

        // Redo radius calc in case the up/ down is pressed.
        /*
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
        */
    }
}
