use std::{collections::HashMap};

use rand::Rng;
use rayon::prelude::*;
use tokio::{sync::{broadcast, watch}, time::Instant};

use super::models::{
        CourseRequest, FitnessCalculator, OptimizationProgress, OptimizedCourse, Particle, PsoParameters, TimePreferenceRequest, PSO
    };

impl Particle {
   
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::rng();
        
      
        let position: Vec<f32> = (0..dimension)
            .map(|_| rng.random_range(0.0..1.0))
            .collect();
            
        let velocity: Vec<f32> = (0..dimension)
            .map(|_| rng.random_range(-0.1..0.1))
            .collect();

        Particle {
            position,
            velocity,
            pbest_position: vec![0.0; dimension], 
            pbest_fitness: f32::INFINITY,        
            fitness: f32::INFINITY,              
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
            
            let cognitive = cognitive_weight * r1 * (self.pbest_position[i] - self.position[i]);
            
            let social = social_weight * r2 * (gbest[i] - self.position[i]);
            
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive + social;
            
            // self.velocity[i] = self.velocity[i].clamp(-1.0, 1.0);
        }
    }

    pub fn update_position(&mut self) {
        for i in 0..self.position.len() {
            self.position[i] += self.velocity[i];

            // self.position[i] = self.position[i].clamp(0.0, 1.0);
        }
    }

    pub fn update_personal_best(&mut self) {
        if self.fitness < self.pbest_fitness && !self.fitness.is_nan() {
            self.pbest_fitness = self.fitness;
            self.pbest_position = self.position.clone();
        }
    }
}


impl PSO {
    pub fn new(
        courses: Vec<CourseRequest>,
        time_preferences: Vec<TimePreferenceRequest>,
        parameters: PsoParameters,
        status_tx: Option<broadcast::Sender<OptimizationProgress>>,
        stop_rx: Option<watch::Receiver<bool>>,
    ) -> Self {
        let dimension = courses.len() * 2; 

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

        self.initialize_swarm();

        for iteration in 0..self.parameters.max_iterations {

            if let Some(rx) = &self.stop_rx {
                if *rx.borrow() {
                    println!("⛔ Optimization stopped at iteration {}", iteration);
                    break;
                }
            }

            self.evaluate_all_particles();

            self.update_global_best();

            self.update_all_particles();

            if self.global_best_fitness < 0.001 {
                println!("Early stopping: Optimal solution found at iteration {}", iteration);
                break;
            }

            self.progress(iteration + 1, &start_time, all_best_fitness, current_run, total_runs, false);

        }

        // Final results
        all_best_fitness.push(self.global_best_fitness);

        println!("Optimization completed - Best fitness: {:.6}", self.global_best_fitness);
        (self.global_best_position.clone(), self.global_best_fitness)
    }

    fn reset_optimization(&mut self) {
        self.global_best_fitness = f32::INFINITY;
        self.global_best_position.fill(0.0);
        self.particles.clear();
    }

    fn initialize_swarm(&mut self) {
        let dimension = self.courses.len() * 2;
        
        self.particles = (0..self.parameters.swarm_size)
            .map(|_| Particle::new(dimension))
            .collect();

        println!("Swarm initialized with {} particles, {} dimensions each", 
                self.parameters.swarm_size, dimension);
    }

    fn evaluate_all_particles(&mut self) {
        let courses = self.courses.clone();
        let fitness_calculator = self.fitness_calculator.clone();

        self.particles.par_iter_mut().for_each(|particle| {
            particle.fitness = Self::evaluate_position(&particle.position, &courses, &fitness_calculator);
            particle.update_personal_best();
        });
    }

    fn update_global_best(&mut self) {
        for particle in &self.particles {
            if particle.pbest_fitness < self.global_best_fitness && !particle.pbest_fitness.is_nan() {
                self.global_best_fitness = particle.pbest_fitness;
                self.global_best_position = particle.pbest_position.clone();
            }
        }
    }

    fn update_all_particles(&mut self) {
        let global_best_position = self.global_best_position.clone();
        let params = self.parameters.clone();

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

    pub fn evaluate_position(
        position: &[f32],
        courses: &[CourseRequest],
        calculator: &FitnessCalculator,
    ) -> f32 {
        let schedule = Self::position_to_schedule(position, courses);
        calculator.calculate_fitness(&schedule)
    }
    
    pub fn position_to_schedule(
        position: &[f32],
        courses: &[CourseRequest],
    ) -> Vec<OptimizedCourse> {
        let mut grouped: HashMap<(u32, u32, u32, u32), Vec<(f32, f32, OptimizedCourse)>> = HashMap::new();

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

        for entries in grouped.into_values() {
            let mut sorted = entries;
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let max_sks = if sorted.len() == 4 { 3 } else { 6 };
            let mut sks_per_day = [0u32; 5]; 
            let mut current_day = 0;

            for (_, time_order, mut course) in sorted {
                while current_day < 5 {
                    if sks_per_day[current_day] + course.sks <= max_sks {
                        course.hari = current_day as u32 + 1;
                        sks_per_day[current_day] += course.sks;
                        break;
                    }
                    current_day += 1;
                }

                if course.hari == 0 {
                    course.hari = 5;
                }

                scheduled.push((
                    (course.prodi, course.semester, course.id_kelas, course.id_waktu, course.hari),
                    time_order,
                    course,
                ));
            }
        }

        let mut by_day: HashMap<_, Vec<_>> = HashMap::new();
        for (key, time_order, course) in scheduled {
            by_day.entry(key).or_default().push((time_order, course));
        }

        let mut final_schedule = Vec::with_capacity(courses.len());

        for ((_, _, _, id_waktu, _), mut entries) in by_day {
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let (start, end) = match id_waktu {
                1 => (480, 720),   
                2 => (1080, 1320), 
                _ => (480, 720),   
            };

            let mut current_time = start;

            for (_, mut course) in entries {
                let duration = course.sks * 40; 
                
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