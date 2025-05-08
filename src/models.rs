use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Deserialize, Clone)]
pub struct CourseRequest {
    pub id_jadwal: u64,
    pub id_matkul: u64,
    pub id_dosen: u64,
    pub id_waktu: u64,
    pub id_kelas: u64,
    pub semester: u64,
    pub sks: u64,
    pub prodi: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TimePreferenceRequest {
    pub id_dosen: u64,
    pub seninPagi: bool,
    pub seninMalam: bool,
    pub selasaPagi: bool,
    pub selasaMalam: bool,
    pub rabuPagi: bool,
    pub rabuMalam: bool,
    pub kamisPagi: bool,
    pub kamisMalam: bool,
    pub jumatPagi: bool,
    pub jumatMalam: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OptimizationRequest {
    pub courses: Vec<CourseRequest>,
    pub parameters: PsoParameters,
    pub time_preferences: Vec<TimePreferenceRequest>,
    pub sum_ruangan: u64,
}

#[derive(Debug, Serialize, Clone)]
pub struct OptimizedCourse {
    pub id_jadwal: u64,
    pub id_matkul: u64,
    pub id_dosen: u64,
    pub id_kelas: u64,
    pub id_waktu: u64,
    pub hari: u64,
    pub jam_mulai: u64,
    pub jam_akhir: u64,
    pub ruangan: u64,
    pub semester: u64,
    pub sks: u64,
    pub prodi: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct PsoParameters {
    pub swarm_size: usize,
    pub max_iterations: usize,
    pub cognitive_weight: f64,
    pub social_weight: f64,
    pub inertia_weight: f64,
    pub velocity_clamp: f64,     // Ganti V_MAX
    pub position_clamp: f64,       // Ganti POS_MIN
}


#[derive(Debug, Clone, Serialize, Default)]
pub struct ConflictInfo {
    pub group_conflicts: Vec<((u64, u64, u64), (u64, u64, u64))>,
    pub preference_conflicts: Vec<u64>,
    pub conflicts_list: Vec<String>,
    pub total_conflicts: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct OptimizationStatus {
    pub iteration: usize,
    pub elapsed_time: Duration,
    // pub current_positions: Vec<Vec<f64>>,
    pub current_fitness: f64,
    pub best_fitness: f64,
    pub is_finished: bool,
    pub conflicts: ConflictInfo,
}
