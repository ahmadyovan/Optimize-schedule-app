use std::collections::BTreeMap;
use std::time::Instant;
use hashbrown::HashMap;
use tokio::sync::{broadcast, watch};
use crate::models::{
    ConflictInfo, CourseRequest, OptimizationStatus, OptimizedCourse, PsoParameters, TimePreferenceRequest
};
use super::particle::Particle;
use super::fitness::FitnessCalculator;

pub struct PSO {
    particles: Vec<Particle>,
    global_best_position: Vec<f64>,
    global_best_fitness: f64,
    parameters: PsoParameters,
    courses: Vec<CourseRequest>,
    fitness_calculator: FitnessCalculator,
    sum_ruangan: u64,
}

impl PSO {
    pub fn new(
        courses: Vec<CourseRequest>,
        time_preferences: Vec<TimePreferenceRequest>,
        parameters: PsoParameters,
        sum_ruangan: u64
    ) -> Self {
        // Each course requires 3 values (day, time slot, room)
        let dimension = courses.len() * 2;
        
        // Create particles based on swarm size from parameters
        let particles = (0..parameters.swarm_size)
            .map(|_| Particle::new(dimension))
            .collect::<Vec<_>>();
        
        // Create fitness calculator
        let fitness_calculator = FitnessCalculator::new(time_preferences);
        
        PSO {
            particles,
            global_best_position: vec![0.0; dimension],
            global_best_fitness: f64::INFINITY,
            courses,
            fitness_calculator,
            parameters,
            sum_ruangan,
        }
    }
    
    pub async fn optimize(
        &mut self,
        status_tx: broadcast::Sender<OptimizationStatus>,
        stop_rx: watch::Receiver<bool>,
    ) -> Vec<f64> {
        let start_time = Instant::now();
        self.initialize_swarm();
        
        let courses = &self.courses;
        let sum_ruangan = self.sum_ruangan;
        let params = &self.parameters;
    
        for iteration in 0..params.max_iterations {
            // Check early stopping signal
            if *stop_rx.borrow() {
                break;
            }
    
            // Update each particle
            for particle in &mut self.particles {
                // Update velocity and position
                particle.update_velocity(
                    &self.global_best_position,
                    params.inertia_weight,
                    params.cognitive_weight,
                    params.social_weight,
                    params.velocity_clamp,
                );
                particle.update_position(params.position_clamp);
    
                // Evaluate fitness
                let schedule = Self::position_to_schedule(&particle.position, courses, sum_ruangan);
                let (fitness, _) = self.fitness_calculator.calculate_fitness(&schedule);
    
                // Update personal best
                if fitness < particle.pbest_fitness {
                    particle.pbest_fitness = fitness;
                    particle.pbest_position = particle.position.clone();
                }
            }
    
            // Find new global best
            if let Some(best_particle) = self.particles.iter().min_by(|a, b| {
                a.pbest_fitness.partial_cmp(&b.pbest_fitness).unwrap()
            }) {
                if best_particle.pbest_fitness < self.global_best_fitness {
                    self.global_best_fitness = best_particle.pbest_fitness;
                    self.global_best_position = best_particle.pbest_position.clone();
                }
            }
    
            // Send status update
            let schedule = Self::position_to_schedule(&self.global_best_position, courses, sum_ruangan);
            let (_, current_conflicts) = self.fitness_calculator.calculate_fitness(&schedule);
    
            let status = OptimizationStatus {
                iteration,
                elapsed_time: start_time.elapsed(),
                current_fitness: self.global_best_fitness,
                best_fitness: self.global_best_fitness,
                is_finished: false,
                conflicts: current_conflicts,
            };
    
            // Non-blocking send, ignore if no receivers
            let _ = status_tx.send(status);
    
            // Yield to prevent blocking the async runtime
            tokio::task::yield_now().await;
        }
    
        // Send final status
        let _ = status_tx.send(OptimizationStatus {
            iteration: params.max_iterations - 1,
            elapsed_time: start_time.elapsed(),
            current_fitness: self.global_best_fitness,
            best_fitness: self.global_best_fitness,
            is_finished: true,
            conflicts: ConflictInfo::default(),
        });
    
        self.global_best_position.clone()
    }
    
    // Initialize all particles and find initial global best
    fn initialize_swarm(&mut self) {
        let courses = &self.courses;
        let sum_ruangan = self.sum_ruangan;
        let fitness_calculator = &self.fitness_calculator;
        
        for particle in &mut self.particles {
            let schedule = Self::position_to_schedule(&particle.position, courses, sum_ruangan);
            let (fitness, _) = fitness_calculator.calculate_fitness(&schedule);
            
            // Update personal best
            if fitness < particle.pbest_fitness {
                particle.pbest_fitness = fitness;
                particle.pbest_position = particle.position.clone();
            }
            
            // Update global best
            if fitness < self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best_position = particle.position.clone();
            }
        }
    }
    
    pub fn position_to_schedule(
        position: &[f64],
        courses: &[CourseRequest],
        sum_ruangan: u64,
    ) -> Vec<OptimizedCourse> {
        let mut schedule_entries = Vec::with_capacity(courses.len());
    
        // Alokasikan ruangan (sekali saja)
        let mut room_allocation = HashMap::new();
        let mut current_room = 1;
        let mut groups: Vec<_> = courses.iter()
            .map(|c| (c.prodi, c.semester, c.id_kelas))
            .collect();
        groups.sort_unstable();
        groups.dedup();
    
        for group in groups {
            room_allocation.insert(group, current_room);
            current_room = (current_room % sum_ruangan) + 1;
        }
    
        // Step 1: Masukkan data awal dari partikel
        for (i, course) in courses.iter().enumerate() {
            let base_idx = i * 2;
            if base_idx + 1 >= position.len() {
                break;
            }
    
            let day_order = position[base_idx];
            let urutan_pelajaran = position[base_idx + 1];
    
            schedule_entries.push((
                (course.prodi, course.semester, course.id_kelas, course.id_waktu),
                day_order,
                urutan_pelajaran,
                OptimizedCourse {
                    id_jadwal: course.id_jadwal,
                    id_matkul: course.id_matkul,
                    id_dosen: course.id_dosen,
                    id_kelas: course.id_kelas,
                    id_waktu: course.id_waktu,
                    hari: 0,
                    jam_mulai: 0,
                    jam_akhir: 0,
                    ruangan: *room_allocation.get(&(course.prodi, course.semester, course.id_kelas)).unwrap_or(&1),
                    semester: course.semester,
                    sks: course.sks,
                    prodi: course.prodi,
                },
            ));
        }
    
        // Step 2: Kelompokkan berdasarkan prodi, semester, kelas, id_waktu
        let mut grouped_by_group: HashMap<(u64, u64, u64, u64), Vec<(f64, f64, OptimizedCourse)>> = HashMap::new();
        for (key, day_order, urutan, course) in schedule_entries {
            grouped_by_group.entry(key).or_default().push((day_order, urutan, course));
        }
    
        let mut with_day_set = Vec::with_capacity(courses.len());
    
        for (_group_key, mut entries) in grouped_by_group {
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
            let mut sks_per_day = [0u64; 5];
            let mut current_day = 0;
    
            for (_, urutan, mut course) in entries {
                while current_day < 5 {
                    if sks_per_day[current_day] + course.sks <= 6 {
                        course.hari = current_day as u64 + 1;
                        sks_per_day[current_day] += course.sks;
                        break;
                    }
                    current_day += 1;
                }
    
                if course.hari == 0 {
                    course.hari = 5;
                }
    
                with_day_set.push((
                    (course.prodi, course.semester, course.id_kelas, course.id_waktu, course.hari),
                    urutan,
                    course,
                ));
            }
        }
    
        // Step 3: Kelompokkan berdasarkan hari
        let mut grouped_by_day: HashMap<(u64, u64, u64, u64, u64), Vec<(f64, OptimizedCourse)>> = HashMap::new();
        for (key, urutan, course) in with_day_set {
            grouped_by_day.entry(key).or_default().push((urutan, course));
        }
    
        let mut final_schedule = Vec::with_capacity(courses.len());
    
        for ((_prodi, _semester, _kelas, id_waktu, _hari), mut entries) in grouped_by_day {
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
            let (start_time, max_time) = match id_waktu {
                1 => (480, 720),
                2 => (1080, 1320),
                _ => (480, 720),
            };
    
            let mut current_time = start_time;
    
            for (_, mut course) in entries {
                let duration = course.sks * 40;
                if current_time + duration <= max_time {
                    course.jam_mulai = current_time;
                    course.jam_akhir = current_time + duration;
                    current_time += duration;
                } else {
                    course.jam_mulai = start_time;
                    course.jam_akhir = start_time + duration;
                    current_time = start_time + duration;
                }
    
                final_schedule.push(course);
            }
        }
    
        final_schedule
    }
    
    
    
    // Method to evaluate best position
    pub fn evaluate_best_position(&self) -> (f64, ConflictInfo) {
        let schedule = Self::position_to_schedule(&self.global_best_position, &self.courses, self.sum_ruangan);
        self.fitness_calculator.calculate_fitness(&schedule)
    }
}