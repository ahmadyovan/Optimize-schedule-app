use std::{
    collections::HashMap, sync::{atomic::{AtomicBool, Ordering}, Arc},
};

use rand::Rng;
use rayon::prelude::*;
use tokio::{sync::{broadcast, watch}, time::Instant};

use super::models::{
        CourseRequest, FitnessCalculator, OptimizationProgress, OptimizedCourse, Particle, PsoParameters, TimePreferenceRequest, PSO
    };

// ============================================================================
// PARTICLE IMPLEMENTATION
// ============================================================================
impl Particle {
    /// Create new particle with random position and velocity
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::rng();
        
        // Random position in [0,1] range
        let position: Vec<f32> = (0..dimension)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
            
        // Small random velocity for stable convergence
        let velocity: Vec<f32> = (0..dimension)
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();

        Particle {
            position,
            velocity,
            pbest_position: vec![0.0; dimension], // Will be set after first evaluation
            pbest_fitness: f32::INFINITY,        // Initialize with infinity
            fitness: f32::INFINITY,              // Will be calculated in first iteration
        }
    }

    /// Update velocity using standard PSO formula
    pub fn update_velocity(
        &mut self,
        gbest: &[f32],
        inertia_weight: f32,
        cognitive_weight: f32,
        social_weight: f32,
    ) {
        let mut rng = rand::rng();
        
        for i in 0..self.velocity.len() {
            let r1: f32 = rng.random(); 
            let r2: f32 = rng.random(); 
            
            // Cognitive component: attraction to personal best
            let cognitive = cognitive_weight * r1 * (self.pbest_position[i] - self.position[i]);
            
            // Social component: attraction to global best
            let social = social_weight * r2 * (gbest[i] - self.position[i]);
            
            // PSO velocity update: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive + social;
            
            // Velocity clamping to prevent excessive exploration
            self.velocity[i] = self.velocity[i].clamp(-0.5, 0.5);
        }
    }

    /// Update position based on velocity
    pub fn update_position(&mut self) {
        for i in 0..self.position.len() {
            self.position[i] += self.velocity[i];
            // Keep position in [0,1] range
            self.position[i] = self.position[i].clamp(0.0, 1.0);
        }
    }

    /// Update personal best if current fitness is better
    pub fn update_personal_best(&mut self) {
        if self.fitness < self.pbest_fitness && !self.fitness.is_nan() {
            self.pbest_fitness = self.fitness;
            self.pbest_position = self.position.clone();
        }
    }
}

// ============================================================================
// PSO IMPLEMENTATION
// ============================================================================
impl PSO {
    /// Constructor for new PSO instance
    pub fn new(
        courses: Vec<CourseRequest>,
        time_preferences: Vec<TimePreferenceRequest>,
        parameters: PsoParameters,
        status_tx: Option<broadcast::Sender<OptimizationProgress>>,
        stop_rx: Option<watch::Receiver<bool>>,
    ) -> Self {
        let dimension = courses.len() * 2; // 2 dimensions per course: day_order, time_order

        PSO {
            particles: vec![],
            global_best_position: vec![0.0; dimension],
            global_best_fitness: f32::INFINITY,
            courses,
            parameters,
            fitness_calculator: FitnessCalculator::new(time_preferences),
            status_tx,
            stop_rx
        }
    }

    /// Main PSO optimization function
    pub async fn optimize(
        &mut self,
        run_info: Option<(usize, usize)>,
        all_best_fitness: &mut Vec<f32>,
    ) -> (Vec<f32>, f32) {
        let start_time = Instant::now();
        let (current_run, total_runs) = run_info.unwrap_or((0, 0));

        // Reset state for new run
        self.reset_optimization();

        println!("Starting PSO optimization - Run {}/{}", current_run + 1, total_runs);
        println!("Swarm size: {}, Max iterations: {}", self.parameters.swarm_size, self.parameters.max_iterations);

        // Initialize swarm with random particles
        self.initialize_swarm();

        // Main optimization loop
        for iteration in 0..self.parameters.max_iterations {

            if let Some(rx) = &self.stop_rx {
                if *rx.borrow() {
                    println!("⛔ Optimization stopped at iteration {}", iteration);
                    break;
                }
            }

            // Step 1: Evaluate all particles
            self.evaluate_all_particles();

            // Step 2: Update global best
            self.update_global_best();

            // Step 3: Update all particles (velocity and position)
            self.update_all_particles();

            // Progress reporting
            // self.emit_progress(window, iteration + 1, &start_time, all_best_fitness, current_run, total_runs, false);

            // Early stopping for very good solutions
            if self.global_best_fitness < 0.001 {
                println!("Early stopping: Optimal solution found at iteration {}", iteration);
                break;
            }

            // Log progress periodically
            if iteration % 100 == 0 {
                let diversity = self.calculate_swarm_diversity();
                println!("Iteration {}: Best fitness = {:.6}, Diversity = {:.6}", 
                    iteration, self.global_best_fitness, diversity);
            }

            self.progress(iteration + 1, &start_time, all_best_fitness, current_run, total_runs, false);

        }

        // Final results
        all_best_fitness.push(self.global_best_fitness);
        // self.emit_progress(window, self.parameters.max_iterations, &start_time, all_best_fitness, current_run, total_runs, true);

        println!("Optimization completed - Best fitness: {:.6}", self.global_best_fitness);
        (self.global_best_position.clone(), self.global_best_fitness)
    }

    /// Reset optimization state for new run
    fn reset_optimization(&mut self) {
        self.global_best_fitness = f32::INFINITY;
        self.global_best_position.fill(0.0);
        self.particles.clear();
    }

    /// Initialize swarm with random particles (no fitness evaluation here)
    fn initialize_swarm(&mut self) {
        let dimension = self.courses.len() * 2;
        
        self.particles = (0..self.parameters.swarm_size)
            .map(|_| Particle::new(dimension))
            .collect();

        println!("Swarm initialized with {} particles, {} dimensions each", 
                self.parameters.swarm_size, dimension);
    }

    /// Evaluate fitness for all particles
    fn evaluate_all_particles(&mut self) {
        // Clone necessary data for parallel processing
        let courses = self.courses.clone();
        let fitness_calculator = self.fitness_calculator.clone();

        // Parallel fitness evaluation
        self.particles.par_iter_mut().for_each(|particle| {
            particle.fitness = Self::evaluate_position(&particle.position, &courses, &fitness_calculator);
            particle.update_personal_best();
        });
    }

    /// Update global best from all particles
    fn update_global_best(&mut self) {
        for particle in &self.particles {
            if particle.pbest_fitness < self.global_best_fitness && !particle.pbest_fitness.is_nan() {
                self.global_best_fitness = particle.pbest_fitness;
                self.global_best_position = particle.pbest_position.clone();
            }
        }
    }

    /// Update all particles (velocity and position)
    fn update_all_particles(&mut self) {
        // Clone global best for parallel access
        let global_best_position = self.global_best_position.clone();
        let params = self.parameters.clone();

        // Parallel particle updates
        self.particles.par_iter_mut().for_each(|particle| {
            particle.update_velocity(
                &global_best_position,
                params.inertia_weight,
                params.cognitive_weight,
                params.social_weight,
            );
            particle.update_position();
        });
    }

    /// Calculate swarm diversity for convergence monitoring
    pub fn calculate_swarm_diversity(&self) -> f32 {
        if self.particles.len() < 2 {
            return 1.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        // Calculate average pairwise distance
        for i in 0..self.particles.len() {
            for j in (i + 1)..self.particles.len() {
                let distance: f32 = self.particles[i].position
                    .iter()
                    .zip(&self.particles[j].position)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                
                total_distance += distance;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / count as f32
        } else {
            1.0
        }
    }

    fn progress(
        &self,
        iteration: usize,
        start_time: &Instant,
        all_best_fitness: &[f32],
        current_run: usize,
        total_runs: usize,
        is_finished: bool,
    ) {
         let progress = OptimizationProgress {
            iteration,
            elapsed_time: start_time.elapsed(),
            all_best_fitness: Some(all_best_fitness.to_vec()),
            best_fitness: self.global_best_fitness,
            current_run: Some(current_run),
            total_runs: Some(total_runs),
            is_finished,
        };

        if let Some(tx) = &self.status_tx {
            let _ = tx.send(progress);
        }
    }

    /// Evaluate fitness of a position
    pub fn evaluate_position(
        position: &[f32],
        courses: &[CourseRequest],
        calculator: &FitnessCalculator,
    ) -> f32 {
        let schedule = Self::position_to_schedule(position, courses);
        calculator.calculate_fitness(&schedule)
    }
    
    /// Convert particle position to valid schedule
    pub fn position_to_schedule(
        position: &[f32],
        courses: &[CourseRequest],
    ) -> Vec<OptimizedCourse> {
        let mut grouped: HashMap<(u32, u32, u32, u32), Vec<(f32, f32, OptimizedCourse)>> = HashMap::new();

        // Group courses by prodi, semester, class, and time
        for (i, course) in courses.iter().enumerate() {
            let idx = i * 2;
            if idx + 1 >= position.len() {
                break;
            }

            let day_order = position[idx];
            let time_order = position[idx + 1];
            let key = (course.prodi, course.semester, course.id_kelas, course.id_waktu);

            let opt_course = OptimizedCourse {
                id_jadwal: course.id_jadwal,
                id_matkul: course.id_matkul,
                id_dosen: course.id_dosen,
                id_kelas: course.id_kelas,
                id_waktu: course.id_waktu,
                hari: 0,
                jam_mulai: 0,
                jam_akhir: 0,
                ruangan: 0,
                semester: course.semester,
                sks: course.sks,
                prodi: course.prodi,
            };

            grouped.entry(key).or_default().push((day_order, time_order, opt_course));
        }

        let mut scheduled = Vec::with_capacity(courses.len());

        // Schedule days based on day_order
        for entries in grouped.into_values() {
            let mut sorted = entries;
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // SKS limit per day based on number of courses
            let max_sks = if sorted.len() == 4 { 3 } else { 6 };
            let mut sks_per_day = [0u32; 5]; // Monday-Friday
            let mut current_day = 0;

            for (_, time_order, mut course) in sorted {
                // Find available day
                while current_day < 5 {
                    if sks_per_day[current_day] + course.sks <= max_sks {
                        course.hari = current_day as u32 + 1; // 1=Monday, 2=Tuesday, etc.
                        sks_per_day[current_day] += course.sks;
                        break;
                    }
                    current_day += 1;
                }

                // Fallback to Friday if no slot available
                if course.hari == 0 {
                    course.hari = 5; // Friday
                }

                scheduled.push((
                    (course.prodi, course.semester, course.id_kelas, course.id_waktu, course.hari),
                    time_order,
                    course,
                ));
            }
        }

        // Schedule times based on time_order
        let mut by_day: HashMap<_, Vec<_>> = HashMap::new();
        for (key, time_order, course) in scheduled {
            by_day.entry(key).or_default().push((time_order, course));
        }

        let mut final_schedule = Vec::with_capacity(courses.len());

        for ((_, _, _, id_waktu, _), mut entries) in by_day {
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Determine time range based on id_waktu
            let (start, end) = match id_waktu {
                1 => (480, 720),   // Morning: 08:00-12:00 (in minutes)
                2 => (1080, 1320), // Evening: 18:00-22:00 (in minutes)
                _ => (480, 720),   // Default morning
            };

            let mut current_time = start;

            for (_, mut course) in entries {
                let duration = course.sks * 40; // 40 minutes per SKS
                
                // Reset to start if not enough time
                if current_time + duration > end {
                    current_time = start;
                }

                course.jam_mulai = current_time;
                course.jam_akhir = current_time + duration;
                current_time += duration;

                final_schedule.push(course);
            }
        }

        final_schedule
    }
}