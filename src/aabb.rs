#[derive(Default, Clone)]
pub struct SBV {
    pub center: [f32; 3],
    pub radius: f32,
}

impl SBV {
    pub fn new(points: &[[f32; 3]]) -> Self {
        let mut center = [0.; 3];
        for p in points {
            center[0] += p[0];
            center[1] += p[1];
            center[2] += p[2];
        }
        center[0] = center[0] / points.len() as f32;
        center[1] = center[1] / points.len() as f32;
        center[2] = center[2] / points.len() as f32;
        let mut radius = 0.;
        for p in points {
            let d_x = p[0] - center[0];
            let d_y = p[1] - center[1];
            let d_z = p[2] - center[2];
            let dist = d_x * d_x + d_y * d_y + d_z * d_z;
            if dist > radius {
                radius = dist;
            }
        }
        radius = radius.sqrt();
        Self { center, radius }
    }

    pub fn merge(box1: &mut Option<SBV>, box2: &SBV) {
        if let Some(box1) = box1 {
            let d_x = box1.center[0] - box2.center[0];
            let d_z = box1.center[2] - box2.center[2];
            let dist = (d_x * d_x + d_z * d_z).sqrt();
            if dist + box1.radius <= box2.radius {
                box1.center = box2.center;
                box1.radius = box2.radius;
            } else if dist + box2.radius <= box1.radius {
                ()
            } else {
                box1.center[0] = (box1.center[0]
                    + box2.center[0]
                    + (box1.center[0] - box2.center[0]) * (box1.radius - box2.radius) / dist)
                    / 2.;
                box1.center[2] = (box1.center[2]
                    + box2.center[2]
                    + (box1.center[2] - box2.center[2]) * (box1.radius - box2.radius) / dist)
                    / 2.;
                box1.radius = 0.5 * (box1.radius + box2.radius + dist);
            }
        } else {
            *box1 = Some(box2.clone());
        }
    }

    pub fn get_bb(&self) -> [f32; 4] {
        [
            self.center[0] - self.radius,
            self.center[0] + self.radius,
            self.center[2] - self.radius,
            self.center[2] + self.radius,
        ]
    }

    pub fn transform(&self, transform: &[[f32; 4]; 4]) -> Self {
        use cgmath::InnerSpace;
        let transform: cgmath::Matrix4<f32> = (*transform).into();
        let center: cgmath::Point3<f32> = self.center.into();
        let center = transform * center.to_homogeneous();
        let center = center / center.w;
        let center = [center.x, center.y, center.z];
        let volume = transform.x.truncate().magnitude().max(
            transform
                .y
                .truncate()
                .magnitude()
                .max(transform.z.truncate().magnitude()),
        );
        let radius = self.radius * volume;
        Self { center, radius }
    }
}
