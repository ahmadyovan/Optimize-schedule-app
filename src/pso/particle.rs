use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub pbest_position: Vec<f64>,
    pub pbest_fitness: f64,
}

impl Particle {
    pub fn new(dimension: usize) -> Self {
        use rand::seq::SliceRandom;

        let mut rng = rand::thread_rng();

        // Bagi domain [0,1) menjadi `dimension` sel
        let mut lhs_values: Vec<f64> = (0..dimension)
            .map(|i| {
                let step = 1.0 / dimension as f64;
                let min = i as f64 * step;
                let max = (i + 1) as f64 * step;
                rng.gen_range(min..max) // pilih secara acak dalam sel ini
            })
            .collect();

        lhs_values.shuffle(&mut rng); // acak urutannya

        // Gunakan hasil LHS sebagai posisi awal
        let position = lhs_values.clone();

        // Velocity tetap bisa acak normal
        let velocity: Vec<f64> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

        Particle {
            position: position.clone(),
            velocity,
            pbest_position: position,
            pbest_fitness: f64::INFINITY,
        }
    }
    
    pub fn update_velocity(&mut self, gbest: &[f64], inertia_weight: f64, cognitive_weight: f64, social_weight: f64, velocity_clamp: f64,) {
        let mut rng = rand::thread_rng();
        const V_MAX: f64 = 2.0; // Batas maksimum velocity (20% dari rentang posisi)
    
        for i in 0..self.velocity.len() {
            let r1 = rng.gen::<f64>();
            let r2 = rng.gen::<f64>();
    
            // Komponen kognitif (pbest - posisi sekarang)
            let cognitive = cognitive_weight * r1 * (self.pbest_position[i] - self.position[i]);
            
            // Komponen sosial (gbest - posisi sekarang)
            let social = social_weight * r2 * (gbest[i] - self.position[i]);
    
            // Update velocity dengan inertia
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive + social;
    
            // Velocity clamping untuk mencegah eksplosi
            self.velocity[i] = self.velocity[i].clamp(-velocity_clamp, velocity_clamp);
        }
    }
    
    pub fn update_position(&mut self, position_clamp: f64) {
        const POS_MIN: f64 = 0.0; // Batas minimum posisi
    
        for i in 0..self.position.len() {
            // Update posisi
            self.position[i] += self.velocity[i];
            
            // Position clamping
            self.position[i] = self.position[i].clamp(POS_MIN, position_clamp);
            
        }
    }
}